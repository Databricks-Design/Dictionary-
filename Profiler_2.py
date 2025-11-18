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

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from packages.spacy_model import SpacyModel
    from triton_python_backend_utils import Tensor
    from tests.mocks import mockInferenceRequest
    from memory_profiler import LineProfiler, show_results
except ImportError as e:
    print(f"Setup Error: {e}")
    sys.exit(1)

# ============================================================================
# CONFIG
# ============================================================================
NUM_TRANSACTIONS = 1000
BATCH_SIZE = 50
MEMORY_THRESHOLD_MB = 0.1
LEAK_REPORT_FILE = "leak_report.txt"

# ============================================================================
# DATA GENERATION
# ============================================================================
def generate_unique_transaction(iteration: int, num_unique_tokens: int = 50) -> str:
    txn_id = f"TXN{iteration:010d}"
    unique_tokens = [f"UNK-{iteration}-{i}" for i in range(num_unique_tokens)]
    return f"{txn_id} {' '.join(unique_tokens)}"

# ============================================================================
# SMART ANALYZER
# ============================================================================
def analyze_profiler_output(output_str):
    significant_lines = []
    current_function = "Unknown"
    
    lines = output_str.split('\n')
    for line in lines:
        if 'def ' in line:
            parts = line.split()
            try:
                current_function = parts[parts.index('def') + 1].split('(')[0]
            except: pass
            continue

        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit() and 'MiB' in line:
            try:
                line_num = int(parts[0])
                mem_usage = float(parts[1])
                increment = float(parts[3])
                code = ' '.join(parts[4:])
                
                # Filter A: Ignore baseline (Increment ~= Mem Usage)
                if abs(mem_usage - increment) < 1.0: continue
                # Filter B: Ignore tiny noise
                if increment < 0.1: continue
                # Filter C: Ignore wrapper noise
                if "site-packages" in line or "memory_profiler.py" in line: continue
                
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
# MAIN RUNNER
# ============================================================================
def main():
    os.environ['DESCRIPTORS_TO_REMOVE'] = 'LLC,PTY,INC'
    
    # Initialize Log File
    with open(LEAK_REPORT_FILE, "w") as f:
        f.write(f"MEMORY INVESTIGATION REPORT - {datetime.now()}\n")
        f.write("==================================================\n\n")

    print(f"--- Starting Investigation (Transactions: {NUM_TRANSACTIONS}) ---")
    
    # 1. Setup Data
    descriptions = [generate_unique_transaction(i) for i in range(NUM_TRANSACTIONS)]
    df = pd.DataFrame({'description': descriptions, 'memo': [""] * NUM_TRANSACTIONS})
    
    # 2. Initialize Model (No @profile in spacy_model.py!)
    ner = SpacyModel()
    ner.initialize({'model_name': 'us_spacy_ner'})
    ner.add_memory_zone = True
    
    # 3. Setup Profiler Dynamically
    lp = LineProfiler()
    lp.add_function(ner.execute)
    lp.add_function(ner.preprocess_input)
    # lp.add_function(ner.extract_results) # Optional
    
    print("\nProcessing batches...")
    
    profiler = psutil.Process(os.getpid())
    initial_mem = profiler.memory_info().rss / 1024 / 1024
    
    for i in range(0, len(df), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        
        batch_df = df.iloc[i:i+BATCH_SIZE]
        desc_vec = np.array(batch_df['description'].tolist(), dtype='|S0').reshape(BATCH_SIZE, 1)
        memo_vec = np.array(batch_df['memo'].tolist(), dtype='|S0').reshape(BATCH_SIZE, 1)
        
        req = [mockInferenceRequest(inputs=[
            Tensor(data=desc_vec, name='description'),
            Tensor(data=memo_vec, name='memo')
        ])]
        
        mem_before = profiler.memory_info().rss / 1024 / 1024
        
        lp.enable()
        ner.execute(req, ner.add_memory_zone)
        lp.disable()
        
        mem_after = profiler.memory_info().rss / 1024 / 1024
        delta = mem_after - mem_before
        
        if delta > MEMORY_THRESHOLD_MB:
            # Get raw string
            s = StringIO()
            show_results(lp, stream=s)
            culprits = analyze_profiler_output(s.getvalue())
            
            # Construct Report Message
            msg = f"🔎 LEAK in Batch {batch_num} | Net Growth: +{delta:.2f} MB\n"
            msg += "   Culprits:\n"
            if culprits:
                for c in culprits:
                    msg += f"     [{c['func']}] Line {c['line']}: +{c['inc']:.2f} MB | {c['code'][:60]}...\n"
            else:
                msg += "     (Spread across small allocations < 0.1 MB)\n"
            msg += "-" * 60
            
            # Print to Console AND File
            print(msg)
            log_to_file(msg)
            
        lp.code_map.clear()
        del req, desc_vec, memo_vec
        gc.collect()

    final_mem = profiler.memory_info().rss / 1024 / 1024
    end_msg = f"\n--- INVESTIGATION COMPLETE ---\nTotal Permanent Growth: {final_mem - initial_mem:.2f} MB"
    print(end_msg)
    log_to_file(end_msg)
    print(f"Report saved to {LEAK_REPORT_FILE}")

if __name__ == '__main__':
    main()
