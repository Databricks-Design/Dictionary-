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

# --- 1. STRICT IMPORTS (No custom mocks) ---
# These MUST exist in your environment
from packages.spacy_model import SpacyModel
from triton_python_backend_utils import Tensor
from tests.mocks import mockInferenceRequest
from unidecode import unidecode

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_TRANSACTIONS = 50000      
BATCH_SIZE = 50               
SAVE_OUTPUTS = True           
OUTPUT_DIR = "./output_investigation"
BATCHES_PER_FILE = 2000       
LEAK_REPORT_FILE = "final_memory_balance_sheet.txt"
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
    
    print(f"--- Starting Cumulative Analysis (Transactions: {NUM_TRANSACTIONS:,}) ---")
    
    # A. Generate Data
    df = generate_dataset(NUM_TRANSACTIONS)
    
    # B. Init Model
    print("Initializing SpaCy model...")
    ner_model = SpacyModel()
    ner_model.initialize({'model_name': 'us_spacy_ner'})
    ner_model.add_memory_zone = True 
    
    # C. START TRACEMALLOC
    # This takes a snapshot of memory BEFORE we start processing
    print("\nTaking BASELINE memory snapshot...")
    gc.collect()
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()
    
    profiler = psutil.Process(os.getpid())
    initial_rss = profiler.memory_info().rss / 1024 / 1024
    
    # File Tracking (Exact Reference Logic)
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
            
            # 1. Status Update (5 Mins)
            if time.time() - last_status_time >= STATUS_INTERVAL_SECONDS:
                curr_rss = profiler.memory_info().rss / 1024 / 1024
                growth = curr_rss - initial_rss
                print(f"[Status] Batch {batch_num}/{total_batches} | Current RSS Growth: +{growth:.2f} MB")
                last_status_time = time.time()

            # 2. Prepare Inputs (Reference Logic)
            batch_df = df.iloc[i:i+BATCH_SIZE].copy()
            descriptions = batch_df['description'].to_list()
            memo = batch_df['memo'].to_list()
            
            descriptions_vec = np.array(descriptions, dtype='|S0').reshape(len(descriptions), 1)
            memos_vec = np.array(memo, dtype='|S0').reshape(len(memo), 1)
            
            # Strict use of imported Mock/Tensor
            requests = [
                mockInferenceRequest(inputs=[
                    Tensor(data=descriptions_vec, name='description'),
                    Tensor(data=memos_vec, name='memo')
                ])
            ]
            
            # 3. Execute
            raw_results = ner_model.execute(requests, ner_model.add_memory_zone)
            
            # 4. Save Outputs (Reference Logic)
            if SAVE_OUTPUTS:
                outputs = []
                for raw_result in raw_results:
                    labels, extracted_texts, entity_ids = raw_result.output_tensors()
                    
                    labels = labels.as_numpy().tolist()
                    extracted_texts = extracted_texts.as_numpy().tolist()
                    entity_ids = entity_ids.as_numpy().tolist()
                    
                    for label_list, extracted_text_list, entity_id_list in zip(labels, extracted_texts, entity_ids):
                        these_outputs = []
                        decoded_labels = [x.decode('utf-8') for x in label_list]
                        decoded_extracted_texts = [x for x in extracted_text_list]
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
            del batch_df, descriptions, memo, requests, raw_results
            if SAVE_OUTPUTS: del outputs
            
            # We do NOT clear tracemalloc. It must accumulate to catch the leak.
            if batch_num % 50 == 0: gc.collect()

    except KeyboardInterrupt:
        print("\nInterrupted. Jumping to Analysis...")

    # ============================================================================
    # 4. GENERATE BALANCE SHEET
    # ============================================================================
    print("\n--- EXECUTION FINISHED ---")
    print("Calculating final memory attribution...")
    gc.collect()
    
    # Take Final Snapshot
    snapshot_end = tracemalloc.take_snapshot()
    final_rss = profiler.memory_info().rss / 1024 / 1024
    
    # Filter for spacy_model.py specifically
    model_filter = [tracemalloc.Filter(inclusive=True, filename_pattern="*spacy_model.py")]
    model_snapshot = snapshot_end.filter_traces(model_filter)
    model_stats = model_snapshot.compare_to(snapshot_start, 'lineno')
    
    # Stats Calculation
    total_python_growth = sum(stat.size_diff for stat in model_stats) / 1024 / 1024
    total_os_growth = final_rss - initial_rss
    unexplained_growth = total_os_growth - total_python_growth
    
    # Print Report
    report = []
    report.append(f"MEMORY BALANCE SHEET - {datetime.now()}")
    report.append("=" * 80)
    report.append(f"A. TOTAL PROCESS GROWTH (RSS):     {total_os_growth:+.2f} MB")
    report.append(f"B. ATTRIBUTED TO SPACY_MODEL.PY:   {total_python_growth:+.2f} MB")
    report.append(f"C. UNATTRIBUTED / C-LEVEL:         {unexplained_growth:+.2f} MB")
    report.append("=" * 80)
    report.append(f"\nDETAILED BREAKDOWN (spacy_model.py)")
    report.append(f"{'FUNCTION':<25} | {'LINE':<5} | {'GROWTH (MB)':<12} | {'ALLOCS':<8} | {'CODE'}")
    report.append("-" * 80)

    has_content = False
    # Show top 20 offending lines
    for stat in model_stats[:20]:
        growth_mb = stat.size_diff / 1024 / 1024
        if growth_mb > 0.001:
            has_content = True
            frame = stat.traceback[0]
            line_num = frame.lineno
            
            # Read the code line
            line_text = linecache.getline(frame.filename, line_num).strip()
            
            # Simple heuristic to find function name
            func_name = "Unknown"
            try:
                with open(frame.filename) as f:
                    lines = f.readlines()
                for i in range(line_num - 1, -1, -1):
                    if lines[i].strip().startswith("def "):
                        func_name = lines[i].strip().split()[1].split('(')[0]
                        break
            except: pass

            report.append(f"{func_name:<25} | {line_num:<5} | {growth_mb:<+12.4f} | {stat.count_diff:<+8} | {line_text[:40]}...")

    if not has_content:
        report.append("No significant persistent object growth detected in spacy_model.py.")

    report_str = "\n".join(report)
    print("\n" + report_str)
    
    with open(LEAK_REPORT_FILE, "w") as f:
        f.write(report_str)
    print(f"\nReport saved to: {LEAK_REPORT_FILE}")

if __name__ == '__main__':
    main()
