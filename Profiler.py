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

# --- Ensure these are in your Python path ---
try:
    from packages.spacy_model import SpacyModel
    from triton_python_backend_utils import Tensor
    from tests.mocks import mockInferenceRequest
except ImportError as e:
    print(f"Error: Could not import necessary modules. Make sure 'packages' and 'tests' are in your sys.path.")
    print(e)
    sys.exit(1)

# --- Import the memory profiler API ---
try:
    from memory_profiler import LineProfiler, show_results
except ImportError:
    print("Error: 'memory-profiler' not found. Please run: pip install memory-profiler")
    sys.exit(1)

# ============================================================================
# CONFIG
# ============================================================================
NUM_TRANSACTIONS = 1000
BATCH_SIZE = 50
MEMORY_THRESHOLD_MB = 0.5  # Min batch-over-batch growth to trigger a report
LINE_THRESHOLD_MB = 0.3    # Min line-by-line growth to highlight in report
LEAK_REPORT_FILE = "leak_report.txt"
PROFILER_OUTPUT_FILE = "profiler_output.txt"

# ============================================================================
# SYNTHETIC DATA
# ============================================================================

def generate_unique_transaction(iteration: int, num_unique_tokens: int = 50) -> str:
    """Generates a transaction string with many unique tokens."""
    transaction_types = ["POS", "ATM", "ONLINE", "TRANSFER", "PAYMENT", "REFUND", "WITHDRAWAL", "DEPOSIT"]
    merchants = ["AMAZON", "WALMART", "STARBUCKS", "SHELL", "MCDONALDS", "TARGET", "COSTCO", "BESTBUY", 
                 "NETFLIX", "UBER", "AIRBNB", "BOOKING", "PAYPAL", "VENMO", "SQUARE", "SPOTIFY", 
                 "APPLE", "GOOGLE", "MICROSOFT", "CHIPOTLE"]
    
    txn_id = f"TXN{iteration:010d}"
    merchant = f"{random.choice(merchants)}{random.randint(1000, 9999)}"
    amount = f"${random.uniform(5.0, 999.99):.2f}"
    card = f"CARD-{random.randint(1000, 9999)}"
    acct = f"ACCT{random.randint(1000000, 9999999)}"
    auth = f"AUTH{random.randint(100000, 999999)}"
    ref = f"REF{iteration}{random.randint(1000, 9999)}"
    merchant_id = f"MID{random.randint(100000, 999999)}"
    terminal = f"T{random.randint(1000, 9999)}"
    batch = f"B{random.randint(100, 999)}"
    trans_type = random.choice(transaction_types)
    
    unique_tokens = []
    remaining = num_unique_tokens - 11
    for i in range(remaining):
        token = f"{random.choice(['LOC', 'ID', 'CODE', 'SEQ'])}{iteration}{i}{random.randint(100, 999)}"
        unique_tokens.append(token)
    
    description_parts = [txn_id, trans_type, merchant, amount, card, acct, auth, ref, merchant_id, terminal, batch] + unique_tokens
    return " ".join(description_parts)

# ============================================================================
# PROFILER ANALYSIS
# ============================================================================

def parse_profiler_line(line):
    """Parse a memory_profiler output line"""
    try:
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0].isdigit():
            line_num = int(parts[0])
            mem_usage = float(parts[1])
            if parts[3] != 'MiB':
                increment = float(parts[3])
            else:
                increment = 0.0
            code = ' '.join(parts[4:]) if len(parts) > 4 else ''
            return line_num, mem_usage, increment, code
    except:
        pass
    return None

def analyze_profiler_output(output_str):
    """Find lines with significant memory increments"""
    leaking_lines = []
    current_function = None
    
    for line in output_str.split('\n'):
        if 'def ' in line: # Simpler check for function name
            try:
                parts = line.strip().split()
                idx = parts.index('def')
                if idx + 1 < len(parts):
                    func_name = parts[idx + 1].split('(')[0]
                    current_function = func_name
            except:
                pass
        
        parsed = parse_profiler_line(line)
        if parsed:
            line_num, mem_usage, increment, code = parsed
            if increment > LINE_THRESHOLD_MB:
                leaking_lines.append({
                    'function': current_function,
                    'line': line_num,
                    'increment': increment,
                    'mem_usage': mem_usage,
                    'code': code.strip()
                })
    
    return leaking_lines

# ============================================================================
# LEAK TRACKING
# ============================================================================

class LeakTracker:
    """Handles writing the summary and raw profiler output to files."""
    def __init__(self, report_file, profiler_file):
        self.report_file = report_file
        self.profiler_file = profiler_file
        self.leaks = []
        
        # Initialize leak report
        with open(self.report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MEMORY LEAK DETECTION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
        
        # Initialize profiler output
        with open(self.profiler_file, 'w') as f:
            f.write("="*90 + "\n")
            f.write("MEMORY PROFILER OUTPUT - DETAILED VIEW\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*90 + "\n\n")
    
    def record_leak(self, batch_num, mem_delta, leaking_lines, profiler_output):
        self.leaks.append({
            'batch': batch_num,
            'mem_delta': mem_delta,
            'lines': leaking_lines
        })
        
        # Write to leak report (parsed summary)
        with open(self.report_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"LEAK #{len(self.leaks)} - Batch {batch_num}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Memory increase: {mem_delta:+.2f} MB\n\n")
            
            if leaking_lines:
                f.write(f"Problematic Lines:\n")
                f.write(f"{'-'*80}\n")
                for item in leaking_lines:
                    f.write(f"  Function: {item['function']}()\n")
                    f.write(f"  Line {item['line']}: {item['code']}\n")
                    f.write(f"  Memory: {item['mem_usage']:.2f} MB (Δ +{item['increment']:.2f} MB)\n")
                    f.write(f"{'-'*80}\n")
            else:
                f.write("No specific line identified (increment < threshold)\n")
        
        # Write to profiler output (raw memory_profiler data)
        with open(self.profiler_file, 'a') as f:
            f.write("\n" + "="*90 + "\n")
            f.write(f"LEAK #{len(self.leaks)} - Batch {batch_num} | Memory Increase: {mem_delta:+.2f} MB\n")
            f.write("="*90 + "\n")
            f.write(profiler_output)
            f.write("\n" + "="*90 + "\n\n")
    
    def finalize(self, initial_mem, final_mem, elapsed_time):
        # Finalize leak report
        with open(self.report_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total leaks: {len(self.leaks)}\n")
            f.write(f"Memory growth: {final_mem - initial_mem:.2f} MB\n")
            f.write(f"Elapsed: {elapsed_time:.2f} seconds\n")
            f.write("="*80 + "\n")
        
        # Finalize profiler output
        with open(self.profiler_file, 'a') as f:
            f.write("\n" + "="*90 + "\n")
            f.write("SESSION SUMMARY\n")
            f.write("="*90 + "\n")
            f.write(f"Total leaks captured: {len(self.leaks)}\n")
            f.write(f"Total memory growth: {final_mem - initial_mem:.2f} MB\n")
            f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
            f.write("="*90 + "\n")

# ============================================================================
# PROFILING TEST
# ============================================================================

def run_test_profiling(ner_model: SpacyModel, df: pd.DataFrame, batch_size: int):
    
    profiler = psutil.Process(os.getpid())
    initial_memory = profiler.memory_info().rss / 1024 / 1024
    
    print("="*80)
    print(f"BASELINE: {initial_memory:.2f} MB")
    print("="*80)
    print()
    
    tracker = LeakTracker(LEAK_REPORT_FILE, PROFILER_OUTPUT_FILE)
    
    # Setup LineProfiler
    lp = LineProfiler()
    lp.add_function(ner_model.execute)
    lp.add_function(ner_model.preprocess_input)
    lp.add_function(ner_model.extract_results)
    
    num_batches = (len(df) + batch_size - 1) // batch_size
    start_time = time.time()
    
    print(f"Processing {num_batches} batches...")
    
    for i in range(0, len(df), batch_size):
        batch_num = (i // batch_size) + 1
        
        batch_df = df.iloc[i:i+batch_size]
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
        
        mem_before = profiler.memory_info().rss / 1024 / 1024
        
        # --- ENABLE PROFILER FOR THIS BATCH ---
        lp.enable()
        # Execute
        raw_results = ner_model.execute(requests, ner_model.add_memory_zone)
        lp.disable()
        # --------------------------------------
        
        mem_after = profiler.memory_info().rss / 1024 / 1024
        delta = mem_after - mem_before
        
        # Detect leak
        if delta > MEMORY_THRESHOLD_MB:
            # Capture profiler output
            buffer = StringIO()
            
            # --- This is the correct API call ---
            show_results(lp, stream=buffer)
            
            profiler_output = buffer.getvalue()
            
            # Analyze for leaking lines
            leaking_lines = analyze_profiler_output(profiler_output)
            
            # --- This is the "smart" print you requested ---
            print(f"🚨 LEAK - Batch {batch_num}: +{delta:.2f} MB")
            if leaking_lines:
                for item in leaking_lines[:3]: # Print top 3 leaks
                    print(f"    → {item['function']}() line {item['line']}: +{item['increment']:.2f} MB")
            else:
                 print(f"    → (No specific line > {LINE_THRESHOLD_MB} MB identified)")
            print() # Add a newline for readability
            
            tracker.record_leak(batch_num, delta, leaking_lines, profiler_output)
        
        # Clear stats for the next batch
        lp.clear() 

        # --- THIS BLOCK IS NOW REMOVED ---
        # if batch_num % 10 == 0:
        #    ... (no more chatty status)
        
        del batch_df, descriptions, memo, requests, raw_results
        gc.collect()
    
    elapsed_time = time.time() - start_time
    final_memory = profiler.memory_info().rss / 1024 / 1024
    
    tracker.finalize(initial_memory, final_memory, elapsed_time)
    
    # --- FINAL SUMMARY ---
    print()
    print("="*80)
    print(f"FINAL: {final_memory:.2f} MB | Growth: +{final_memory - initial_memory:.2f} MB")
    print(f"Leaks: {len(tracker.leaks)} | Time: {elapsed_time:.2f}s")
    print(f"\nReports saved:")
    print(f"  - {LEAK_REPORT_FILE} (parsed summary)")
    print(f"  - {PROFILER_OUTPUT_FILE} (raw profiler output)")
    print("="*80)

# ============================================================================
# MAIN
# ============================================================================

def main():
    os.environ['DESCRIPTORS_TO_REMOVE'] = 'LLC,PTY,INC'
    
    print("="*80)
    print("MEMORY LEAK DETECTOR")
    print("="*80)
    print(f"Transactions: {NUM_TRANSACTIONS:,} | Batch: {BATCH_SIZE}")
    print(f"Thresholds: Memory={MEMORY_THRESHOLD_MB} MB | Line={LINE_THRESHOLD_MB} MB")
    print("="*80)
    print()
    
    # Generate data
    print("Generating data...")
    descriptions = [generate_unique_transaction(i) for i in range(NUM_TRANSACTIONS)]
    df = pd.DataFrame({'description': descriptions, 'memo': [""] * NUM_TRANSACTIONS})
    print(f"Generated {len(df):,} transactions\n")
    
    # Initialize model
    print("Initializing model...")
    ner = SpacyModel()
    ner.initialize({'model_name': 'us_spacy_ner'})
    ner.add_memory_zone = True # Test with memory zone
    print("Ready!\n")
    
    # Run
    run_test_profiling(ner, df, BATCH_SIZE)


if __name__ == '__main__':
    main()
