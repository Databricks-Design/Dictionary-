import os
import sys
import numpy as np
import pandas as pd
import time
import gc
import psutil
import random
import glob
from datetime import datetime
from io import StringIO
from collections import defaultdict

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 1. IMPORTS ---
try:
    from packages.spacy_model import SpacyModel
    # Assuming reference utils exist
    from triton_python_backend_utils import Tensor
    from tests.mocks import mockInferenceRequest
    from unidecode import unidecode
    # Profiler
    from memory_profiler import LineProfiler, show_results
except ImportError as e:
    print(f"Setup Error: {e}")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Data
NUM_TRANSACTIONS = 50000      
BATCH_SIZE = 50               

# Outputs
SAVE_OUTPUTS = True           
OUTPUT_DIR = "./output_investigation"
BATCHES_PER_FILE = 2000       

# Leak Detection & Logs
MEMORY_THRESHOLD_MB = 0.1     
LEAK_REPORT_FILE = "leak_summary_report.txt" # Human readable summary
RAW_LOG_PREFIX = "raw_profiler_logs"         # Prefix for raw data files
MAX_LOG_BYTES = 50 * 1024 * 1024             # Rotate log after 50 MB
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
# 3. POST-MORTEM ANALYZER (Handles Multiple Log Files)
# ============================================================================
def run_post_mortem_analysis():
    """Reads ALL rotated raw log files and prints the definitive summary."""
    print("\n" + "="*60)
    print("STARTING POST-MORTEM ANALYSIS")
    print("="*60)
    
    # Find all log parts: raw_profiler_logs_001.txt, _002.txt, etc.
    log_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{RAW_LOG_PREFIX}_*.txt")))
    
    if not log_files:
        print("No raw logs found. (No leaks > threshold were detected).")
        return

    print(f"Analyzing {len(log_files)} log files...")
    
    line_stats = defaultdict(lambda: {'total_growth': 0.0, 'count': 0, 'code': ''})
    leak_events = 0
    current_function = None
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Count Batches
                if line.startswith("=== BATCH"):
                    leak_events += 1
                    continue

                # Track Function Context
                if line.startswith('def '):
                    try: current_function = line.split()[1].split('(')[0]
                    except: pass
                    continue

                # Parse Profiler Lines
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit() and 'MiB' in line:
                    try:
                        line_num = int(parts[0])
                        mem_usage = float(parts[1])
                        increment = float(parts[3])
                        code = ' '.join(parts[4:])
                        
                        # --- SMART FILTERS ---
                        if abs(mem_usage - increment) < 1.0: continue # Baseline
                        if increment < 0.05: continue # Noise
                        if "memory_profiler.py" in line: continue # Wrapper

                        # Aggregate
                        key = (current_function, line_num)
                        line_stats[key]['total_growth'] += increment
                        line_stats[key]['count'] += 1
                        line_stats[key]['code'] = code
                    except: continue

    # --- PRINT TABLE ---
    print(f"TOTAL LEAK EVENTS ANALYZED: {leak_events}")
    print(f"\n🏆 ROOT CAUSE ANALYSIS (Top Offenders)")
    print(f"{'-'*90}")
    print(f"{'FUNCTION':<25} | {'LINE':<5} | {'TOTAL MB':<10} | {'FREQ':<5} | {'CODE'}")
    print(f"{'-'*90}")
    
    # Sort by Total Growth
    sorted_stats = sorted(line_stats.items(), key=lambda x: x[1]['total_growth'], reverse=True)
    
    for (func, line_num), data in sorted_stats[:15]: 
        code_snippet = data['code'][:40] + "..." if len(data['code']) > 40 else data['code']
        print(f"{func:<25} | {line_num:<5} | {data['total_growth']:<10.2f} | {data['count']:<5} | {code_snippet}")
    
    print(f"{'-'*90}\n")
    print(f"Full raw data preserved in directory: {OUTPUT_DIR}")

# ============================================================================
# 4. UTILS
# ============================================================================
def analyze_profiler_output_string(output_str):
    """Helper for the real-time console alert (single batch)."""
    significant_lines = []
    current_function = "Unknown"
    for line in output_str.split('\n'):
        if 'def ' in line:
            try: current_function = line.split()[1].split('(')[0]
            except: pass
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit() and 'MiB' in line:
            try:
                line_num = int(parts[0])
                mem_usage = float(parts[1])
                increment = float(parts[3])
                code = ' '.join(parts[4:])
                if abs(mem_usage - increment) < 1.0: continue
                if increment < 0.05: continue
                if "memory_profiler.py" in line: continue
                significant_lines.append({'func': current_function, 'line': line_num, 'inc': increment, 'code': code})
            except: continue
    return significant_lines

def log_summary(message):
    with open(LEAK_REPORT_FILE, "a") as f:
        f.write(message + "\n")

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
def main():
    if SAVE_OUTPUTS: os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ['DESCRIPTORS_TO_REMOVE'] = 'LLC,PTY,INC'
    
    # Initialize Summary Report
    with open(LEAK_REPORT_FILE, "w") as f:
        f.write(f"MEMORY INVESTIGATION SUMMARY - {datetime.now()}\n\n")

    print(f"--- Starting Investigation (Transactions: {NUM_TRANSACTIONS:,}) ---")
    
    # A. GENERATE DATA
    df = generate_dataset(NUM_TRANSACTIONS)
    
    # B. INIT MODEL
    print("Initializing SpaCy model...")
    ner_model = SpacyModel()
    ner_model.initialize({'model_name': 'us_spacy_ner'})
    ner_model.add_memory_zone = True 
    
    # C. SETUP
    profiler = psutil.Process(os.getpid())
    initial_mem = profiler.memory_info().rss / 1024 / 1024
    
    # File Rotation Setup
    file_counter = 1
    batches_in_current_file = 0
    first_batch_in_file = True
    current_csv_path = os.path.join(OUTPUT_DIR, f'output_part_{file_counter:03d}.csv')
    
    # Raw Log Rotation Setup
    log_counter = 1
    current_raw_log_path = os.path.join(OUTPUT_DIR, f'{RAW_LOG_PREFIX}_{log_counter:03d}.txt')
    # Initialize first log file
    with open(current_raw_log_path, "w") as f:
        f.write(f"RAW LOGS PART {log_counter}\n")

    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    last_status_time = time.time()

    print(f"\nProcessing {total_batches} batches...")
    print(f"Status updates every 5 minutes.")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # --- D. ROBUST EXECUTION LOOP ---
    try:
        for i in range(0, len(df), BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            
            # 1. Status Update
            if time.time() - last_status_time >= STATUS_INTERVAL_SECONDS:
                print(f"[Status] Batch {batch_num}/{total_batches} ({datetime.now().strftime('%H:%M:%S')})")
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
            
            # 3. Profiled Execution
            lp = LineProfiler()
            lp.add_function(ner_model.execute)
            lp.add_function(ner_model.preprocess_input)
            lp.add_function(ner_model.extract_results)
            
            mem_before = profiler.memory_info().rss / 1024 / 1024
            
            lp.enable()
            raw_results = ner_model.execute(requests=req, use_memory_zone=ner_model.add_memory_zone)
            lp.disable()
            
            mem_after = profiler.memory_info().rss / 1024 / 1024
            delta = mem_after - mem_before
            
            # 4. Detect & Log (With Rotation)
            if delta > MEMORY_THRESHOLD_MB:
                # Capture Output
                capture = StringIO()
                old_out = sys.stdout
                sys.stdout = capture
                try: show_results(lp)
                except: pass
                sys.stdout = old_out
                
                raw_text = capture.getvalue()
                
                # Check Log File Size & Rotate if needed
                if os.path.exists(current_raw_log_path) and os.path.getsize(current_raw_log_path) > MAX_LOG_BYTES:
                    log_counter += 1
                    current_raw_log_path = os.path.join(OUTPUT_DIR, f'{RAW_LOG_PREFIX}_{log_counter:03d}.txt')
                    with open(current_raw_log_path, "w") as f:
                        f.write(f"RAW LOGS PART {log_counter}\n")
                
                # Append to current log
                with open(current_raw_log_path, "a") as f:
                    f.write(f"\n=== BATCH {batch_num} | GROWTH: +{delta:.2f} MB ===\n")
                    f.write(raw_text)
                    f.write("=== END BATCH ===\n")

                # Console Alert (Summary only)
                culprits = analyze_profiler_output_string(raw_text)
                msg = f"🔎 LEAK in Batch {batch_num} | Net: +{delta:.2f} MB | "
                if culprits:
                    top = culprits[0]
                    msg += f"Top: {top['func']}() line {top['line']} (+{top['inc']:.2f} MB)"
                else:
                    msg += "Small allocations"
                print(msg)
                log_summary(msg)

            # 5. Save Outputs (With Rotation)
            if SAVE_OUTPUTS:
                outputs = []
                for rr in raw_results:
                    labels, texts, ids = rr.output_tensors()
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

            # 6. Cleanup
            del batch_df, descriptions, memo, req, raw_results, lp
            if SAVE_OUTPUTS: del outputs
            gc.collect()

    except Exception as e:
        print(f"\n!!! EXECUTION CRASHED !!! Error: {e}")
        print("Proceeding to Post-Mortem Analysis to save collected data...")
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Proceeding to Analysis...")

    # ============================================================================
    # 6. FINAL REPORT (Guaranteed to run)
    # ============================================================================
    final_mem = profiler.memory_info().rss / 1024 / 1024
    print(f"\nExecution Finished. Total Process Growth: {final_mem - initial_mem:.2f} MB")
    
    # Run the analyzer on whatever log files exist
    run_post_mortem_analysis()

if __name__ == '__main__':
    main()
