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

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 1. IMPORTS & MOCKS ---
try:
    from packages.spacy_model import SpacyModel
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
SAVE_OUTPUTS = False          
OUTPUT_DIR = "./output_investigation"
LEAK_REPORT_FILE = "global_memory_dump.txt"
STATUS_INTERVAL_SECONDS = 300 

# ============================================================================
# 2. DATA GENERATION
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

# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================
def main():
    if SAVE_OUTPUTS: os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ['DESCRIPTORS_TO_REMOVE'] = 'LLC,PTY,INC'
    
    print(f"--- Starting GLOBAL Deep Scan (Transactions: {NUM_TRANSACTIONS:,}) ---")
    
    # A. Generate Data
    df = generate_dataset(NUM_TRANSACTIONS)
    
    # B. Init Model
    print("Initializing SpaCy model...")
    ner_model = SpacyModel()
    ner_model.initialize({'model_name': 'us_spacy_ner'})
    ner_model.add_memory_zone = True 
    
    # C. BASELINE SNAPSHOT (Global)
    print("\nTaking BASELINE memory snapshot...")
    gc.collect()
    tracemalloc.start()
    tracemalloc.start(25) # Capture 25 frames of stack trace
    snapshot_start = tracemalloc.take_snapshot()
    
    profiler = psutil.Process(os.getpid())
    initial_rss = profiler.memory_info().rss / 1024 / 1024
    
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    last_status_time = time.time()

    print(f"Processing {total_batches} batches...")

    try:
        for i in range(0, len(df), BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            
            # 1. Status Update
            if time.time() - last_status_time >= STATUS_INTERVAL_SECONDS:
                curr_rss = profiler.memory_info().rss / 1024 / 1024
                growth = curr_rss - initial_rss
                print(f"[Status] Batch {batch_num}/{total_batches} | Current RSS Growth: +{growth:.2f} MB")
                last_status_time = time.time()

            # 2. Prepare Inputs
            batch_df = df.iloc[i:i+BATCH_SIZE].copy()
            descriptions = batch_df['description'].to_list()
            memo = batch_df['memo'].to_list()
            
            descriptions_vec = np.array(descriptions, dtype='|S0').reshape(len(descriptions), 1)
            memos_vec = np.array(memo, dtype='|S0').reshape(len(memo), 1)
            
            req = [mockInferenceRequest(inputs=[
                Tensor(data=descriptions_vec, name='description'),
                Tensor(data=memos_vec, name='memo')
            ])]
            
            # 3. Execute
            # This runs the code that is causing the problem.
            raw_results = ner_model.execute(req, ner_model.add_memory_zone)
            
            # 4. Cleanup
            del batch_df, descriptions, memo, req, raw_results
            if batch_num % 50 == 0: gc.collect()

    except KeyboardInterrupt:
        print("\nInterrupted. Jumping to Analysis...")
    except Exception as e:
        print(f"\nExecution CRASHED: {e}. Analyzing partial data...")

    # ============================================================================
    # 4. GLOBAL DEEP SCAN REPORT
    # ============================================================================
    print("\n--- EXECUTION FINISHED ---")
    print("Calculating GLOBAL memory stats (All Files, All Imports)...")
    gc.collect()
    
    snapshot_end = tracemalloc.take_snapshot()
    final_rss = profiler.memory_info().rss / 1024 / 1024
    total_os_growth = final_rss - initial_rss
    
    # Compare ALL traces
    top_stats = snapshot_end.compare_to(snapshot_start, 'traceback')
    
    report = []
    report.append(f"GLOBAL MEMORY DUMP - {datetime.now()}")
    report.append("=" * 100)
    report.append(f"TOTAL PROCESS GROWTH (RSS): {total_os_growth:+.2f} MB")
    report.append("=" * 100)
    report.append(f"Top 10 Offending Tracebacks (Origin of Permanent Allocations):")
    report.append("-" * 100)
    
    total_explained = 0.0
    
    # Show Top 10 Tracebacks (The exact call chain that led to the permanent leak)
    for stat in top_stats[:10]:
        growth_mb = stat.size_diff / 1024 / 1024
        if growth_mb < 0.01: continue # Skip negligible traces
        
        total_explained += growth_mb
        
        report.append(f"\n[ GROWTH: {growth_mb:+.4f} MB ] (Objects: {stat.count_diff})")
        
        # Print the call stack leading up to the leak
        for frame in stat.traceback:
            filename = frame.filename
            line_num = frame.lineno
            line_text = linecache.getline(filename, line_num).strip()
            
            # Identify which library/project the file belongs to
            if 'site-packages/spacy/' in filename:
                library = 'SPACY LIBRARY'
            elif current_dir in filename:
                library = 'YOUR PROJECT'
            else:
                library = 'OTHER LIBRARY'

            report.append(f"  -> {library:<15} {os.path.basename(filename)}:{line_num:<4} {line_text[:60]}")

    report.append("=" * 100)
    report.append(f"Total Growth Explained by Python Objects: {total_explained:.2f} MB")
    report.append(f"Unexplained (C-Level/Fragmentation):      {total_os_growth - total_explained:.2f} MB")

    report_str = "\n".join(report)
    print("\n" + report_str)
    
    with open(LEAK_REPORT_FILE, "w") as f:
        f.write(report_str)
    print(f"\nGlobal Dump saved to: {LEAK_REPORT_FILE}")

if __name__ == '__main__':
    main()
