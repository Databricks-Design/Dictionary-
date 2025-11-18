import os
import sys
import numpy as np
import pandas as pd
import time
import gc
import psutil
import random
import tracemalloc
import linecache
from datetime import datetime
from collections import defaultdict

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 1. IMPORTS & MOCKS ---
try:
    from packages.spacy_model import SpacyModel
    from spacy.tokens import Doc, Token, Span, Lexeme # Needed for object counting
    
    # Robust Mocks
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

except ImportError as e:
    print(f"Setup Error: {e}")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_TRANSACTIONS = 50000 
BATCH_SIZE = 50
OUTPUT_FILE = "language_py_diagnostic.txt"
STATUS_INTERVAL_SECONDS = 300

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_unique_transaction(iteration: int, num_unique_tokens: int = 50) -> str:
    """Generates unique transaction to stress memory."""
    txn_id = f"TXN{iteration:010d}"
    merchant = f"MERCHANT{iteration}"
    unique_tokens = [f"UNK-{iteration}-{i}" for i in range(10)]
    parts = [txn_id, "POS", merchant] + unique_tokens
    return " ".join(parts)

def generate_dataset(num_rows):
    print(f"Generating {num_rows:,} synthetic transactions...")
    descriptions = [generate_unique_transaction(i) for i in range(num_rows)]
    memos = [""] * num_rows
    return pd.DataFrame({'description': descriptions, 'memo': memos})

def get_spacy_object_counts():
    """Count actual spaCy objects in memory."""
    gc.collect()
    counts = {'Doc': 0, 'Token': 0, 'Span': 0, 'Lexeme': 0}
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Doc): counts['Doc'] += 1
            elif isinstance(obj, Token): counts['Token'] += 1
            elif isinstance(obj, Span): counts['Span'] += 1
            elif isinstance(obj, Lexeme): counts['Lexeme'] += 1
        except: pass
    return counts

# ============================================================================
# MAIN DIAGNOSTIC
# ============================================================================

def main():
    report = []
    
    # Initialize
    print("Initializing SpaCy model...")
    os.environ['DESCRIPTORS_TO_REMOVE'] = 'LLC,PTY,INC'
    
    ner_model = SpacyModel()
    ner_model.initialize({'model_name': 'us_spacy_ner'})
    ner_model.add_memory_zone = True
    
    print("Generating test data...")
    df = generate_dataset(NUM_TRANSACTIONS)
    
    # Get process handle
    process = psutil.Process(os.getpid())
    
    # === BASELINE MEASUREMENTS (Start) ===
    print("\n--- Taking Baseline Measurements ---")
    gc.collect()
    
    baseline_rss = process.memory_info().rss / 1024 / 1024
    baseline_objects = get_spacy_object_counts()
    
    print("Starting tracemalloc with 25 frame depth...")
    tracemalloc.start(25)
    snapshot_baseline = tracemalloc.take_snapshot()
    
    # === PROCESSING LOOP ===
    print(f"\n--- Processing {NUM_TRANSACTIONS} transactions ---")
    
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    last_status_time = time.time()
    
    try:
        for i in range(0, len(df), BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            
            # 1. Status Update
            if time.time() - last_status_time >= STATUS_INTERVAL_SECONDS:
                curr_rss = process.memory_info().rss / 1024 / 1024
                growth = curr_rss - baseline_rss
                print(f"[Status] Batch {batch_num}/{total_batches} | Current RSS Growth: +{growth:.2f} MB")
                last_status_time = time.time()
            
            # Prepare batch
            batch_df = df.iloc[i:i+BATCH_SIZE].copy()
            descriptions = batch_df['description'].to_list()
            memos = batch_df['memo'].to_list()
            
            descriptions_vec = np.array(descriptions, dtype='|S0').reshape(len(descriptions), 1)
            memos_vec = np.array(memos, dtype='|S0').reshape(len(memos), 1)
            
            req = [mockInferenceRequest(inputs=[
                Tensor(data=descriptions_vec, name='description'),
                Tensor(data=memos_vec, name='memo')
            ])]
            
            # EXECUTE
            raw_results = ner_model.execute(req, ner_model.add_memory_zone)
            
            # Cleanup
            del batch_df, descriptions, memos, req, raw_results
            if batch_num % 10 == 0:
                 gc.collect()
            
    except KeyboardInterrupt:
        print("\nInterrupted. Proceeding to final analysis...")
    except Exception as e:
        print(f"\nCRITICAL ERROR during execution: {e}. Analyzing partial data...")

    # === FINAL ANALYSIS ===
    print("\n--- Taking Final Measurements ---")
    gc.collect()
    
    final_rss = process.memory_info().rss / 1024 / 1024
    final_objects = get_spacy_object_counts()
    
    snapshot_final = tracemalloc.take_snapshot()
    
    # --- REPORT GENERATION ---
    report.append(f"\n{'='*80}")
    report.append("MEMORY ATTRIBUTION VERDICT (POST-EXECUTION)")
    report.append(f"{'='*80}")
    
    # 1. RSS and Object Deltas
    report.append(f"RSS Memory: {final_rss:.2f} MB (Δ = {final_rss - baseline_rss:+.2f} MB)")
    report.append(f"Live Doc Objects: {final_objects['Doc']} (Δ = {final_objects['Doc'] - baseline_objects['Doc']:+d})")
    report.append(f"Live Token Objects: {final_objects['Token']} (Δ = {final_objects['Token'] - baseline_objects['Token']:+d})")
    
    # 2. Tracemalloc Comparison
    report.append(f"\n{'='*80}")
    report.append("TOP OBJECT GROWTH (Allocations still alive)")
    report.append(f"{'='*80}")
    
    top_diffs = snapshot_final.compare_to(snapshot_baseline, 'traceback')
    
    # Show Top 15 Allocations Globally
    for idx, stat in enumerate(top_diffs[:15], 1):
        size_diff_mb = stat.size_diff / 1024 / 1024
        if size_diff_mb < 0.01: continue
        
        report.append(f"\n#{idx} GROWTH: {size_diff_mb:+.4f} MB | Blocks: {stat.count_diff:+d}")
        report.append("Call Stack (Newest Allocation Line First):")
        
        # Print the call stack leading up to the leak
        for frame_idx, frame in enumerate(stat.traceback[:5]): # Show top 5 frames
            filename = frame.filename
            line_num = frame.lineno
            line_text = linecache.getline(filename, line_num).strip()
            
            if 'site-packages/spacy/' in filename:
                source = 'SPACY'
            elif current_dir in filename:
                source = 'YOUR_CODE'
            else:
                source = 'OTHER_LIB'
            
            indent = "  " * frame_idx
            report.append(f"{indent}[{source}] {os.path.basename(filename)}:{line_num} | {line_text}")

    # --- VERDICT ---
    report.append(f"\n{'='*80}")
    report.append("VERDICT: ROOT CAUSE SUMMARY")
    report.append(f"{'='*80}")
    
    if final_objects['Doc'] - baseline_objects['Doc'] > 100:
        report.append("!!! CRITICAL: Doc objects are leaking. This means a component is holding Doc references.")
    else:
        report.append("✓ Leak is NOT in Doc objects. The growth is likely the Tokenizer cache or Python object overhead.")
    
    
    # --- SAVE REPORT ---
    report_str = "\n".join(report)
    print("\n" + report_str)
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(report_str)
    
    print(f"\nFull diagnostic saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
