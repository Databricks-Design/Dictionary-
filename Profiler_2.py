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

# --- 1. IMPORTS (Matching your reference script) ---
try:
    from packages.spacy_model import SpacyModel
    from triton_python_backend_utils import Tensor
    from tests.mocks import mockInferenceRequest
    from unidecode import unidecode
    
    # Profiler Import
    from memory_profiler import LineProfiler, show_results
except ImportError as e:
    print(f"Setup Error: {e}")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Data Generation
NUM_TRANSACTIONS = 50000      # Total rows to generate
BATCH_SIZE = 50               # Keep small for granular leak detection

# Output Handling
OUTPUT_DIR = "./output_investigation"
BATCHES_PER_FILE = 2000       # Rotate CSV after this many batches

# Leak Detection
MEMORY_THRESHOLD_MB = 0.1     # Only report leaks larger than this
LEAK_REPORT_FILE = "leak_report.txt"
STATUS_INTERVAL_SECONDS = 300 # 5 Minutes

# ============================================================================
# 2. DATA GENERATION
# ============================================================================
def generate_unique_transaction(iteration: int, num_unique_tokens: int = 50) -> str:
    """Generates unique transaction to stress Vocab growth."""
    transaction_types = ["POS", "ATM", "ONLINE", "TRANSFER", "PAYMENT"]
    merchants = ["AMAZON", "WALMART", "STARBUCKS", "UBER", "APPLE"]
    
    txn_id = f"TXN{iteration:010d}"
    merchant = f"{random.choice(merchants)}{iteration}"
    amount = f"${(iteration % 995) + 5.0:.2f}"
    
    unique_tokens = [f"UNK-{iteration}-{i}" for i in range(10)]
    parts = [txn_id, "POS", merchant, amount] + unique_tokens
    return " ".join(parts)

def generate_dataset(num_rows):
    print(f"Generating {num_rows:,} synthetic transactions...")
    descriptions = [generate_unique_transaction(i) for i in range(num_rows)]
    memos = [""] * num_rows
    # Return DF exactly as your script expects
    return pd.DataFrame({'description': descriptions, 'memo': memos})

# ============================================================================
# 3. PROFILER UTILS
# ============================================================================
def analyze_profiler_output(output_str):
    """Parses raw profiler string to find lines with memory growth."""
    significant_lines = []
    current_function = "Unknown"
    
    for line in output_str.split('\n'):
        if 'def ' in line:
            parts = line.split()
            try: current_function = parts[parts.index('def') + 1].split('(')[0]
            except: pass
            continue

        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit() and 'MiB' in line:
            try:
                line_num = int(parts[0])
                mem_usage = float(parts[1])
                increment = float(parts[3])
                code = ' '.join(parts[4:])
                
                # Filters to remove noise
                if abs(mem_usage - increment) < 1.0: continue
                if increment < 0.05: continue
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
# 4. MAIN EXECUTION
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ['DESCRIPTORS_TO_REMOVE'] = 'LLC,PTY,INC'
    
    # Initialize Report
    with open(LEAK_REPORT_FILE, "w") as f:
        f.write(f"MEMORY INVESTIGATION REPORT - {datetime.now()}\n")
        f.write("==================================================\n\n")

    print(f"--- Starting Investigation (Transactions: {NUM_TRANSACTIONS:,}) ---")
    
    # A. GENERATE DATA
    df = generate_dataset(NUM_TRANSACTIONS)
    
    # B. INIT MODEL
    print("Initializing SpaCy model...")
    ner_model = SpacyModel()
    ner_model.initialize({'model_name': 'us_spacy_ner'})
    ner_model.add_memory_zone = True # Set the flag we want to test
    
    # C. SETUP PROFILER
    lp = LineProfiler()
    lp.add_function(ner_model.execute)
    lp.add_function(ner_model.preprocess_input)
    lp.add_function(ner_model.extract_results)
    
    # D. TRACKING VARIABLES
    profiler = psutil.Process(os.getpid())
    initial_mem = profiler.memory_info().rss / 1024 / 1024
    leak_stats = defaultdict(lambda: {'total_inc': 0.0, 'count': 0, 'code': ''})
    
    # File Rotation Tracking (From your reference script)
    file_counter = 1
    batches_in_current_file = 0
    first_batch_in_file = True
    current_csv_path = os.path.join(OUTPUT_DIR, f'output_part_{file_counter:03d}.csv')
    
    num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    last_status_time = time.time()

    print(f"\nProcessing {num_batches} batches...")
    print(f"Outputs will be saved to {OUTPUT_DIR}")
    print(f"Logs in {LEAK_REPORT_FILE}\n")

    for i in range(0, len(df), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        
        # 1. STATUS UPDATE (Every 5 Mins)
        if time.time() - last_status_time >= STATUS_INTERVAL_SECONDS:
            print(f"[Status] Batch {batch_num}/{num_batches} ({datetime.now().strftime('%H:%M:%S')})")
            last_status_time = time.time()

        # 2. PREPARE INPUTS (Reference Logic)
        batch_df = df.iloc[i:i+BATCH_SIZE].copy()
        descriptions = batch_df['description'].to_list()
        memo = batch_df['memo'].to_list()
        
        descriptions_vec = np.array(descriptions, dtype='|S0').reshape(len(descriptions), 1)
        memos_vec = np.array(memo, dtype='|S0').reshape(len(memo), 1)
        
        requests = [
            mockInferenceRequest(inputs=[
                Tensor(data=descriptions_vec, name='description'),
                Tensor(data=memos_vec, name='memo')
            ])
        ]
        
        # 3. EXECUTE & PROFILE
        mem_before = profiler.memory_info().rss / 1024 / 1024
        
        lp.enable()
        raw_results = ner_model.execute(requests, ner_model.add_memory_zone)
        lp.disable()
        
        mem_after = profiler.memory_info().rss / 1024 / 1024
        delta = mem_after - mem_before
        
        # 4. DETECT LEAK
        if delta > MEMORY_THRESHOLD_MB:
            s = StringIO()
            show_results(lp, stream=s)
            culprits = analyze_profiler_output(s.getvalue())
            
            msg = f"🔎 LEAK in Batch {batch_num} | Net Growth: +{delta:.2f} MB\n"
            msg += "   Culprits:\n"
            found_culprit = False
            if culprits:
                for c in culprits:
                    msg += f"     [{c['func']}] Line {c['line']}: +{c['inc']:.2f} MB | {c['code'][:60]}...\n"
                    k = (c['func'], c['line'])
                    leak_stats[k]['total_inc'] += c['inc']
                    leak_stats[k]['count'] += 1
                    leak_stats[k]['code'] = c['code'].strip()
                    found_culprit = True
            
            if not found_culprit: msg += "     (Growth spread across small allocations)\n"
            msg += "-" * 60
            
            print(msg)
            log_to_file(msg)

        # 5. PROCESS OUTPUTS (Reference Logic)
        outputs = []
        for raw_result in raw_results:
            # Use function call () as in your reference
            labels, extracted_texts, entity_ids = raw_result.output_tensors()
            
            labels = labels.as_numpy().tolist()
            extracted_texts = extracted_texts.as_numpy().tolist()
            entity_ids = entity_ids.as_numpy().tolist()
            
            for label_list, extracted_entity_list, entity_id_list in zip(labels, extracted_texts, entity_ids):
                these_outputs = []
                decoded_labels = [x.decode('utf-8') for x in label_list]
                decoded_extracted_texts = [x for x in extracted_entity_list]
                decoded_entity_ids = [x.decode('utf-8') for x in entity_id_list]
                
                for label, extracted_text, entity_id in zip(decoded_labels, decoded_extracted_texts, decoded_entity_ids):
                    if label != '' and label is not None:
                        these_outputs.append({
                            'entity_type': label,
                            'extracted_entity': extracted_text,
                            'standardized_entity': entity_id
                        })
                outputs.append(these_outputs)
        
        batch_df['outputs_ner'] = outputs

        # 6. SAVE CSV (Reference File Rotation Logic)
        if first_batch_in_file:
            batch_df.to_csv(current_csv_path, index=False, mode='w')
            first_batch_in_file = False
        else:
            batch_df.to_csv(current_csv_path, index=False, mode='a', header=False)
        
        batches_in_current_file += 1
        
        if batches_in_current_file >= BATCHES_PER_FILE:
            # Rotate file
            file_counter += 1
            batches_in_current_file = 0
            first_batch_in_file = True
            current_csv_path = os.path.join(OUTPUT_DIR, f'output_part_{file_counter:03d}.csv')

        # 7. CLEANUP
        lp.code_map.clear()
        del batch_df, descriptions, memo, requests, raw_results, outputs
        gc.collect()

    # ============================================================================
    # 5. FINAL SUMMARY
    # ============================================================================
    final_mem = profiler.memory_info().rss / 1024 / 1024
    
    summary = f"\n--- INVESTIGATION COMPLETE ---\n"
    summary += f"Total Permanent Growth: {final_mem - initial_mem:.2f} MB\n\n"
    summary += "🏆 TOP MEMORY OFFENDERS:\n" + "="*60 + "\n"
    
    sorted_stats = sorted(leak_stats.items(), key=lambda x: x[1]['total_inc'], reverse=True)
    for (func, line), data in sorted_stats[:10]:
        summary += f"Function: {func}() | Line: {line}\n  Impact: {data['total_inc']:.2f} MB | Count: {data['count']}\n  Code: {data['code']}\n" + "-"*60 + "\n"

    print(summary)
    log_to_file(summary)
    print(f"Report saved to {LEAK_REPORT_FILE}")

if __name__ == '__main__':
    main()
