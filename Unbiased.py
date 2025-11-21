import os
import sys
import numpy as np
import pandas as pd
import gc
import psutil
import ctypes
import time
from collections import Counter

# --- 1. SETUP ---
try:
    from packages.spacy_model import SpacyModel
    # Standard Mocks
    class Tensor:
        def __init__(self, data, name):
            self.data = data
            self.name = name
        def as_numpy(self): return self.data
    class Request:
        def __init__(self, inputs):
            self.inputs = {t.name: t for t in inputs}
        def get_input_tensor_by_name(self, name): return self.inputs.get(name)
    def mockInferenceRequest(inputs): return Request(inputs)
    
    # Try to import Doc for counting
    import spacy.tokens
    from spacy.tokens import Doc
except ImportError as e:
    print(f"Setup Error: {e}")
    sys.exit(1)

# Load libc for malloc_trim
try:
    libc = ctypes.CDLL("libc.so.6")
except:
    libc = None
    print("WARNING: libc not found. malloc_trim will not work (Are you on Linux?)")

INPUT_CSV_PATH = ""
BATCH_SIZE = 50
process = psutil.Process(os.getpid())

# --- HELPER FUNCTIONS ---
def get_rss_mb():
    gc.collect()
    return process.memory_info().rss / 1024 / 1024

def count_objects():
    """Counts active SpaCy objects in memory"""
    gc.collect()
    counts = Counter()
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Doc):
                counts['Doc'] += 1
            # Check for Thinc/PyTorch tensors if possible
            # (Requires knowing the exact class, generic check below)
            elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'): # Simple array-like check
                counts['Tensor/Array'] += 1
        except:
            pass
    return counts

def run_batch(model, df_subset):
    print(f"\n>>> PROCESSING {len(df_subset)} ROWS...")
    for i in range(0, len(df_subset), BATCH_SIZE):
        batch_df = df_subset.iloc[i:i+BATCH_SIZE].copy()
        descriptions = batch_df['description'].to_list()
        memos = [""] * len(descriptions)
        
        d_vec = np.array(descriptions, dtype='|S0').reshape(len(descriptions), 1)
        m_vec = np.array(memos, dtype='|S0').reshape(len(memos), 1)
        
        req = [mockInferenceRequest(inputs=[
            Tensor(data=d_vec, name='description'),
            Tensor(data=m_vec, name='memo')
        ])]
        
        _ = model.execute(req, use_memory_zone=True)
        del batch_df, descriptions, req

# --- MAIN ---
def main():
    print("--- UNBIASED MEMORY DIAGNOSTIC ---")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: File not found: {INPUT_CSV_PATH}")
        return

    # 1. INIT
    print("1. Initializing...")
    model = SpacyModel()
    model.initialize({'model_name': 'us_spacy_ner'})
    model.add_memory_zone = True
    
    full_df = pd.read_csv(INPUT_CSV_PATH)
    df_load = full_df.iloc[:250000] # Large load
    
    gc.collect()
    baseline_rss = get_rss_mb()
    baseline_objs = count_objects()
    print(f"   Baseline RSS: {baseline_rss:.2f} MB")
    print(f"   Baseline Docs: {baseline_objs['Doc']}")

    # 2. STRESS TEST
    run_batch(model, df_load)
    
    after_load_rss = get_rss_mb()
    growth = after_load_rss - baseline_rss
    print(f"\n2. Post-Load RSS: {after_load_rss:.2f} MB (Growth: +{growth:.2f} MB)")
    
    # 3. THE TRUTH TEST: Force Reclaim
    print("\n>>> ATTEMPTING FORCED TRIM (malloc_trim)...")
    if libc:
        libc.malloc_trim(0)
        print("   Trim command sent.")
    else:
        print("   Skipped (No libc).")
        
    final_rss = get_rss_mb()
    reclaimed = after_load_rss - final_rss
    remaining_growth = final_rss - baseline_rss
    
    print(f"\n3. Final Status:")
    print(f"   RSS After Trim: {final_rss:.2f} MB")
    print(f"   Reclaimed:      {reclaimed:.2f} MB")
    print(f"   Remaining Leak: {remaining_growth:.2f} MB")
    
    # 4. DIAGNOSIS
    print("\n" + "="*60)
    print("DIAGNOSTIC VERDICT")
    print("="*60)
    
    if reclaimed > (growth * 0.7):
        print("✅ DIAGNOSIS: FRAGMENTATION (CONFIRMED)")
        print("   - The memory WAS free, but the OS was holding it.")
        print("   - malloc_trim() successfully forced it back.")
        print("   - Conclusion: memory_zone works. Input Masking is the correct fix to prevent fragmentation.")
        
    elif remaining_growth > 50:
        print("❌ DIAGNOSIS: ACTIVE LEAK DETECTED")
        print("   - Forced trim FAILED to reclaim memory.")
        print("   - This means the memory is actively HELD by an object.")
        
        print("\n   >>> INVESTIGATING ACTIVE OBJECTS...")
        final_objs = count_objects()
        doc_growth = final_objs['Doc'] - baseline_objs['Doc']
        tensor_growth = final_objs['Tensor/Array'] - baseline_objs['Tensor/Array']
        
        print(f"   - Active 'Doc' Objects Growth:   {doc_growth}")
        print(f"   - Active 'Tensor' Arrays Growth: {tensor_growth}")
        
        # Check StringStore size manually
        vocab_len = len(model.nlp.vocab)
        strings_len = len(model.nlp.vocab.strings)
        print(f"   - Current Vocab Size: {vocab_len}")
        print(f"   - Current StringStore: {strings_len}")
        
        if doc_growth > 100:
            print("\n   !!! ROOT CAUSE FOUND: Doc Object Leak")
            print("   Docs are not being deleted. Check if they are stored in a global list.")
        elif strings_len > 10000:
             print("\n   !!! ROOT CAUSE FOUND: StringStore Not Reset")
             print("   memory_zone did NOT clear the strings logically.")
        else:
             print("\n   !!! ROOT CAUSE: Hidden C++ State")
             print("   Docs are gone. Strings are gone. Yet memory is held.")
             print("   This points to internal model buffers (e.g. Transformer/Tok2Vec caches).")
             
    else:
        print("❓ RESULT: Inconclusive / Minor Growth")

if __name__ == '__main__':
    main()
