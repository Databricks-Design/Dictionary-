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
SAVE_OUTPUTS = True           
OUTPUT_DIR = "./output_investigation"
BATCHES_PER_FILE = 2000       
LEAK_REPORT_FILE = "final_3layer_balance_sheet.txt"
STATUS_INTERVAL_SECONDS = 300 

# ============================================================================
# 2. DATA GENERATION
# ============================================================================
def generate_unique_transaction(iteration: int, num_unique_tokens: int = 50) -> str:
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
    
    print(f"--- Starting 3-Layer Analysis (Transactions: {NUM_TRANSACTIONS:,}) ---")
    
    # A. Generate Data
    df = generate_dataset(NUM_TRANSACTIONS)
    
    # B. Init Model
    print("Initializing SpaCy model...")
    ner_model = SpacyModel()
    ner_model.initialize({'model_name': 'us_spacy_ner'})
    ner_model.add_memory_zone = True 
    
    # C. BASELINE SNAPSHOT
    print("\nTaking BASELINE memory snapshot...")
    gc.collect()
    
    # Start Tracing
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()
    
    # Start OS Measurement
    profiler = psutil.Process(os.getpid())
    initial_rss = profiler.memory_info().rss / 1024 / 1024
    
    # File Tracking
    file_counter = 1
    batches_in_current_file = 0
    first_batch_in_file = True
    current_csv_path = os.path.join(OUTPUT_DIR, f'output_part_{file_counter:03d}.csv')
    
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

            # 2. Inputs
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
            raw_results = ner_model.execute(req, ner_model.add_memory_zone)
            
            # 4. Outputs
            if SAVE_OUTPUTS:
                outputs = []
                for rr in raw_results:
                    labels, extracted_texts, entity_ids = rr.output_tensors()
                    labels = labels.as_numpy().tolist()
                    texts = texts.as_numpy().tolist()
                    ids = ids.as_numpy().tolist()
                    
                    for l_l, t_l, i_l in zip(labels, texts, ids):
                        row_res = []
                        l_dec = [x.decode('utf-8') if isinstance(x, bytes) else x for x in l_l]
                        t_dec = [x.decode('utf-8') if isinstance(x, bytes) else x for x in t_l]
                        for lbl, txt in zip(l_dec, t_dec):
                            if lbl: row_res.append(f"{lbl}:{txt}")
                        outputs.append("; ".join(row_res))
                
                batch_df['outputs_ner'] = outputs

                if first_batch_in_file:
                    batch_df.to_csv(current_csv_path, index=False, mode='w')
                    first_batch_in_file = False
                else:
                    batch_df.to_csv(current_csv_path, index=False, mode='a', header=False)
                
                batches_in_current_file += 1
                if batches_in_current_file >= BATCHES_PER_FILE:
                    file_counter += 1
                    batches_in_current_file = 0
                    first_batch_in_file = True
                    current_csv_path = os.path.join(OUTPUT_DIR, f'output_part_{file_counter:03d}.csv')

            # 5. Cleanup
            del batch_df, descriptions, memo, req, raw_results
            if SAVE_OUTPUTS: del outputs
            if batch_num % 50 == 0: gc.collect()

    except KeyboardInterrupt:
        print("\nInterrupted. Jumping to Analysis...")

    # ============================================================================
    # 4. THE 3-LAYER BALANCE SHEET
    # ============================================================================
    print("\n--- EXECUTION FINISHED ---")
    print("Calculating final 3-layer memory attribution...")
    gc.collect()
    
    snapshot_end = tracemalloc.take_snapshot()
    
    # A. TOTAL OS GROWTH
    final_rss = profiler.memory_info().rss / 1024 / 1024
    total_os_growth = final_rss - initial_rss
    
    # B. TOTAL PYTHON GROWTH (All code)
    all_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
    total_python_growth = sum(stat.size_diff for stat in all_stats) / 1024 / 1024
    
    # C. USER CODE GROWTH (Project Directory Filter)
    # Filter: Include anything in the current project folder, exclude site-packages
    def is_user_code(filename):
        return current_dir in filename and "site-packages" not in filename
    
    # Manually filter the stats to be precise
    user_code_stats = []
    other_code_stats = []
    user_code_growth = 0.0
    
    for stat in all_stats:
        frame = stat.traceback[0]
        growth_mb = stat.size_diff / 1024 / 1024
        if is_user_code(frame.filename):
            user_code_stats.append(stat)
            user_code_growth += growth_mb
        else:
            other_code_stats.append(stat)

    # D. OTHER PYTHON GROWTH
    other_python_growth = total_python_growth - user_code_growth
    
    # E. TRUE C-LEVEL GROWTH
    true_c_level_growth = total_os_growth - total_python_growth
    
    # --- GENERATE REPORT ---
    report = []
    report.append(f"3-LAYER MEMORY BALANCE SHEET - {datetime.now()}")
    report.append("=" * 90)
    report.append(f"{'CATEGORY':<40} | {'GROWTH (MB)':<15} | {'% OF TOTAL'}")
    report.append("-" * 90)
    pct_os = 100.0
    report.append(f"{'1. TOTAL PROCESS GROWTH (RSS)':<40} | {total_os_growth:<+15.2f} | {pct_os:.1f}%")
    report.append("-" * 90)
    
    pct_py = (total_python_growth / total_os_growth * 100) if total_os_growth > 0 else 0
    pct_user = (user_code_growth / total_os_growth * 100) if total_os_growth > 0 else 0
    pct_other = (other_python_growth / total_os_growth * 100) if total_os_growth > 0 else 0
    pct_c = (true_c_level_growth / total_os_growth * 100) if total_os_growth > 0 else 0

    report.append(f"{'2. PYTHON OBJECTS (Total)':<40} | {total_python_growth:<+15.2f} | {pct_py:.1f}%")
    report.append(f"{'   a. YOUR CODE (Project Folder)':<40} | {user_code_growth:<+15.2f} | {pct_user:.1f}%")
    report.append(f"{'   b. EXTERNAL LIBS (Pandas, etc)':<40} | {other_python_growth:<+15.2f} | {pct_other:.1f}%")
    report.append("-" * 90)
    report.append(f"{'3. UNATTRIBUTED / C-LEVEL':<40} | {true_c_level_growth:<+15.2f} | {pct_c:.1f}%")
    report.append("   (Fragmentation, Thinc/Blis C++ arrays, Malloc overhead)")
    report.append("=" * 90)
    
    # Detailed Breakdown 1: YOUR CODE
    report.append(f"\n\n=== TOP OFFENDERS: YOUR CODE ({current_dir}) ===")
    if not user_code_stats:
        report.append("  (No significant growth detected)")
    else:
        # Sort and slice
        user_code_stats.sort(key=lambda x: x.size_diff, reverse=True)
        for stat in user_code_stats[:15]:
            growth_mb = stat.size_diff / 1024 / 1024
            if growth_mb > 0.001:
                line_desc = str(stat.traceback[0]).replace(current_dir, ".")
                report.append(f"  {growth_mb:+.4f} MB | {line_desc}")

    # Detailed Breakdown 2: OTHER SCRIPTS
    report.append(f"\n\n=== TOP OFFENDERS: EXTERNAL LIBRARIES ===")
    other_code_stats.sort(key=lambda x: x.size_diff, reverse=True)
    for stat in other_code_stats[:10]:
        growth_mb = stat.size_diff / 1024 / 1024
        if growth_mb > 0.01:
            line_desc = str(stat.traceback[0])
            # Shorten site-packages path for readability
            if "site-packages" in line_desc:
                line_desc = ".../" + line_desc.split("site-packages/")[-1]
            report.append(f"  {growth_mb:+.4f} MB | {line_desc}")

    report_str = "\n".join(report)
    print("\n" + report_str)
    
    with open(LEAK_REPORT_FILE, "w") as f:
        f.write(report_str)
    print(f"\nReport saved to: {LEAK_REPORT_FILE}")

if __name__ == '__main__':
    main()
