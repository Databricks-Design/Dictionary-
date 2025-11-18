import os
import sys
import numpy as np
import pandas as pd
import time
import gc
import psutil
import random
from datetime import datetime
from io import StringIO
from collections import defaultdict

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- IMPORTS & MOCKS ---
try:
    from packages.spacy_model import SpacyModel
    # Mock Objects for local execution
    try:
        from triton_python_backend_utils import Tensor
        from tests.mocks import mockInferenceRequest, Request
    except ImportError:
        # Fallback Dummy Classes if mocks aren't found
        class Tensor:
            def __init__(self, data, name):
                self.data = data
                self.name = name
            def as_numpy(self):
                return self.data
        class Request:
            def __init__(self, inputs):
                self.inputs = {t.name: t for t in inputs}
            def get_input_tensor_by_name(self, name):
                return self.inputs.get(name)
                
    # Memory Profiler Import
    from memory_profiler import LineProfiler, show_results

except ImportError as e:
    print(f"Setup Error: {e}")
    print("Please ensure 'memory-profiler' is installed and 'packages/spacy_model.py' exists.")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_TRANSACTIONS = 50000      # Total records to process
BATCH_SIZE = 50               # Small batch size for granular leak detection
MEMORY_THRESHOLD_MB = 0.1     # Report leaks larger than this
STATUS_INTERVAL_SECONDS = 300 # Print "Processing..." every 5 minutes

# Output Settings
SAVE_OUTPUTS = True           # Set False to skip CSV writing (Faster)
BATCHES_PER_FILE = 2000       # New CSV file every 2000 batches (100k records)
OUTPUT_DIR = "./output_investigation"
LEAK_REPORT_FILE = "leak_report.txt"

# ============================================================================
# 1. DATA GENERATION (Realistic)
# ============================================================================
def generate_unique_transaction(iteration: int, num_unique_tokens: int = 50) -> str:
    """Generates unique transaction to stress Vocab growth."""
    transaction_types = ["POS", "ATM", "ONLINE", "TRANSFER", "PAYMENT", "REFUND"]
    merchants = ["AMAZON", "WALMART", "STARBUCKS", "UBER", "APPLE", "GOOGLE"]
    
    txn_id = f"TXN{iteration:010d}"
    merchant = f"{random.choice(merchants)}{iteration}" # Unique merchant string
    amount = f"${(iteration % 995) + 5.0:.2f}"
    
    # Add unique noise tokens to force Vocab entries
    unique_tokens = [f"UNK-{iteration}-{i}" for i in range(10)] 
    
    parts = [txn_id, "POS", merchant, amount] + unique_tokens
    return " ".join(parts)

def generate_dataset(num_rows):
    print(f"Generating {num_rows:,} synthetic transactions...")
    descriptions = [generate_unique_transaction(i) for i in range(num_rows)]
    memos = [""] * num_rows
    return pd.DataFrame({'description': descriptions, 'memo': memos})

# ============================================================================
# 2. PROFILER ANALYSIS UTILS
# ============================================================================
def analyze_profiler_output(output_str):
    """Parses raw profiler string to find lines with memory growth."""
    significant_lines = []
    current_function = "Unknown"
    
    for line in output_str.split('\n'):
        # Track function context
        if 'def ' in line:
            parts = line.split()
            try:
                current_function = parts[parts.index('def') + 1].split('(')[0]
            except: pass
            continue

        # Parse metrics
        parts = line.split()
        # Line format: Line #, Mem usage, Increment, Code
        if len(parts) >= 4 and parts[0].isdigit() and 'MiB' in line:
            try:
                line_num = int(parts[0])
                mem_usage = float(parts[1])
                increment = float(parts[3])
                code = ' '.join(parts[4:])
                
                # --- SMART FILTERS ---
                # 1. Ignore Baseline (First line of function often shows total usage as increment)
                if abs(mem_usage - increment) < 1.0: continue
                # 2. Ignore tiny noise
                if increment < 0.05: continue
                # 3. Ignore wrapper overhead
                if "memory_profiler.py" in line: continue
                
                significant_lines.append({
                    'func': current_function,
                    'line': line_num,
                    'inc': increment,
                    'code': code
                })
            except ValueError: continue

    return significant_lines

def log_to_file(message):
    with open(LEAK_REPORT_FILE, "a") as f:
        f.write(message + "\n")

# ============================================================================
# 3. MAIN INVESTIGATION LOOP
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ['DESCRIPTORS_TO_REMOVE'] = 'LLC,PTY,INC'
    
    # Initialize Report
    with open(LEAK_REPORT_FILE, "w") as f:
        f.write(f"MEMORY INVESTIGATION REPORT - {datetime.now()}\n")
        f.write("==================================================\n\n")

    print(f"--- Starting Investigation ---")
    print(f"Transactions: {NUM_TRANSACTIONS:,} | Batch Size: {BATCH_SIZE}")
    print(f"Saving Outputs: {SAVE_OUTPUTS}")
    
    # --- A. GENERATE DATA ---
    df = generate_dataset(NUM_TRANSACTIONS)
    
    # --- B. INIT MODEL ---
    print("Initializing SpaCy model...")
    ner = SpacyModel()
    ner.initialize({'model_name': 'us_spacy_ner', 'model_version': '1'})
    ner.add_memory_zone = True
    
    # --- C. SETUP PROFILER ---
    # Dynamically attach to spacy_model functions. 
    # DO NOT use @profile in spacy_model.py
    lp = LineProfiler()
    lp.add_function(ner.execute)
    lp.add_function(ner.preprocess_input)
    lp.add_function(ner.extract_results)
    # Add any other specific internal functions if needed
    
    # --- D. TRACKING VARIABLES ---
    profiler = psutil.Process(os.getpid())
    initial_mem = profiler.memory_info().rss / 1024 / 1024
    
    # Stats: Key=(Function, Line), Value={total_inc, count, code}
    leak_stats = defaultdict(lambda: {'total_inc': 0.0, 'count': 0, 'code': ''})
    
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    last_status_time = time.time()
    
    # Output CSV tracking
    current_file_index = 1
    batches_in_current_file = 0
    current_csv_path = os.path.join(OUTPUT_DIR, f"output_part_{current_file_index:03d}.csv")
    is_new_file = True

    print(f"\nProcessing {total_batches} batches...")
    
    for i in range(0, len(df), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        
        # 1. STATUS UPDATE (Every 5 Mins)
        if time.time() - last_status_time >= STATUS_INTERVAL_SECONDS:
            print(f"[Status] Batch {batch_num}/{total_batches} | Time: {datetime.now().strftime('%H:%M:%S')}")
            last_status_time = time.time()

        # 2. PREPARE INPUTS
        batch_df = df.iloc[i:i+BATCH_SIZE].copy()
        desc_vec = np.array(batch_df['description'].tolist(), dtype='|S0').reshape(len(batch_df), 1)
        memo_vec = np.array(batch_df['memo'].tolist(), dtype='|S0').reshape(len(batch_df), 1)
        
        # Create Request (Mock or Real)
        if 'Request' in globals():
            req = [Request(inputs=[Tensor(desc_vec, 'description'), Tensor(memo_vec, 'memo')])]
        else:
            req = [mockInferenceRequest(inputs=[Tensor(desc_vec, 'description'), Tensor(memo_vec, 'memo')])]
        
        # 3. EXECUTION & PROFILING
        mem_before = profiler.memory_info().rss / 1024 / 1024
        
        # Only profile the model execution to save time/noise
        lp.enable()
        raw_results = ner.execute(req, ner.add_memory_zone)
        lp.disable()
        
        mem_after = profiler.memory_info().rss / 1024 / 1024
        delta = mem_after - mem_before
        
        # 4. LEAK DETECTION
        if delta > MEMORY_THRESHOLD_MB:
            # Analyze
            s = StringIO()
            show_results(lp, stream=s)
            culprits = analyze_profiler_output(s.getvalue())
            
            # Report
            msg = f"🔎 LEAK in Batch {batch_num} | Net Growth: +{delta:.2f} MB\n"
            msg += "   Culprits:\n"
            
            found_culprit = False
            if culprits:
                for c in culprits:
                    msg += f"     [{c['func']}] Line {c['line']}: +{c['inc']:.2f} MB | {c['code'][:60]}...\n"
                    
                    # Update Stats
                    k = (c['func'], c['line'])
                    leak_stats[k]['total_inc'] += c['inc']
                    leak_stats[k]['count'] += 1
                    leak_stats[k]['code'] = c['code'].strip()
                    found_culprit = True
            
            if not found_culprit:
                msg += "     (Growth spread across small allocations < 0.05 MB)\n"
            msg += "-" * 60
            
            print(msg)
            log_to_file(msg)
            
        # 5. OUTPUT SAVING (Optional)
        if SAVE_OUTPUTS:
            # Parse results similar to your run_test logic
            outputs = []
            for rr in raw_results:
                # Assuming output_tensors structure from mock/real response
                # Logic tailored to standard mock structure
                t_map = {t.name(): t.as_numpy() for t in rr.output_tensors()} if hasattr(rr, 'output_tensors') else {t.name: t.as_numpy() for t in rr.output_tensors}
                
                labels = t_map.get('label', []).tolist()
                texts = t_map.get('extractedText', []).tolist()
                ids = t_map.get('entityId', []).tolist()
                
                # Flatten batch results
                for l_list, t_list, id_list in zip(labels, texts, ids):
                    row_res = []
                    # Decode bytes if necessary
                    l_dec = [x.decode('utf-8') if isinstance(x, bytes) else x for x in l_list]
                    t_dec = [x.decode('utf-8') if isinstance(x, bytes) else x for x in t_list]
                    id_dec = [x.decode('utf-8') if isinstance(x, bytes) else x for x in id_list]
                    
                    for lbl, txt, eid in zip(l_dec, t_dec, id_dec):
                        if lbl: row_res.append(f"{lbl}:{txt}")
                    outputs.append("; ".join(row_res))

            batch_df['ner_output'] = outputs
            
            # Write to CSV
            mode = 'w' if is_new_file else 'a'
            header = is_new_file
            batch_df.to_csv(current_csv_path, index=False, mode=mode, header=header)
            is_new_file = False
            batches_in_current_file += 1
            
            # Rotate file check
            if batches_in_current_file >= BATCHES_PER_FILE:
                current_file_index += 1
                current_csv_path = os.path.join(OUTPUT_DIR, f"output_part_{current_file_index:03d}.csv")
                batches_in_current_file = 0
                is_new_file = True

        # 6. CLEANUP
        lp.code_map.clear() # IMPORTANT: Reset profiler stats for next batch
        del batch_df, desc_vec, memo_vec, req, raw_results
        gc.collect()

    # ============================================================================
    # 4. FINAL SUMMARY
    # ============================================================================
    final_mem = profiler.memory_info().rss / 1024 / 1024
    
    summary = f"\n--- INVESTIGATION COMPLETE ---\n"
    summary += f"Total Permanent Growth: {final_mem - initial_mem:.2f} MB\n\n"
    summary += "🏆 TOP MEMORY OFFENDERS (Lines causing permanent growth):\n"
    summary += "="*60 + "\n"
    
    # Sort by Total Growth
    sorted_stats = sorted(leak_stats.items(), key=lambda x: x[1]['total_inc'], reverse=True)
    
    if sorted_stats:
        for (func, line), data in sorted_stats[:10]: # Top 10
            summary += f"Function: {func}() | Line: {line}\n"
            summary += f"  Total Impact: {data['total_inc']:.2f} MB\n"
            summary += f"  Frequency:    {data['count']} batches\n"
            summary += f"  Code:         {data['code']}\n"
            summary += "-"*60 + "\n"
    else:
        summary += "No specific lines consistently exceeded threshold.\n"

    print(summary)
    log_to_file(summary)
    print(f"Report saved to {LEAK_REPORT_FILE}")
    if SAVE_OUTPUTS:
        print(f"Outputs saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
