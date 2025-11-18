import os
import psutil
import gc
import uuid
import random
import pandas as pd
import matplotlib.pyplot as plt
import tracemalloc

def get_rss_mb():
    """Gets the current Resident Set Size (RSS) memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# --- CONFIGURATION ---
TOTAL_BATCHES = 10000
BATCH_SIZE = 50
LOG_INTERVAL = 20
DUPLICATE_PERCENTAGE = 25  # Change to 50, 75, etc.
OUTPUT_FILENAME = f'case3_case4_duplicate_{DUPLICATE_PERCENTAGE}pct.png'

# Calculate how many duplicates per batch
NUM_DUPLICATES = int(BATCH_SIZE * DUPLICATE_PERCENTAGE / 100)
NUM_UNIQUE = BATCH_SIZE - NUM_DUPLICATES

print(f"Batch size: {BATCH_SIZE}, Duplicates: {NUM_DUPLICATES} ({DUPLICATE_PERCENTAGE}%), Unique: {NUM_UNIQUE}")

# --- DATA COLLECTION ---
data_case3 = []  # Add with duplicates
data_case4 = []  # Add with duplicates then evict

# --- START ---
print("="*70)
print(f"CASE 3: ADD STRINGS WITH {DUPLICATE_PERCENTAGE}% DUPLICATES")
print("="*70)
gc.collect()
baseline_rss = get_rss_mb()
print(f"Baseline RSS: {baseline_rss:.2f} MB\n")

tracemalloc.start()

# CASE 3: Add strings with duplicates
global_word_cache = {}
reusable_pool = []  # Pool of strings to reuse as duplicates
last_rss = baseline_rss
step_threshold_mb = 50

for i in range(TOTAL_BATCHES):
    # Create batch with duplicates
    new_batch = []
    
    # Add unique strings
    for _ in range(NUM_UNIQUE):
        new_string = str(uuid.uuid4())
        new_batch.append(new_string)
        reusable_pool.append(new_string)  # Add to pool for future reuse
    
    # Add duplicate strings (from pool)
    if len(reusable_pool) >= NUM_DUPLICATES:
        duplicates = random.sample(reusable_pool, NUM_DUPLICATES)
        new_batch.extend(duplicates)
    else:
        # Not enough pool yet, add more unique
        for _ in range(NUM_DUPLICATES):
            new_batch.append(str(uuid.uuid4()))
    
    # Add to cache
    for word in new_batch:
        if word not in global_word_cache:
            global_word_cache[word] = True
    
    # Log at intervals
    if i % LOG_INTERVAL == 0:
        rss_now = get_rss_mb()
        dict_size = len(global_word_cache)
        current_traced, peak_traced = tracemalloc.get_traced_memory()
        traced_mb = current_traced / (1024 * 1024)
        
        rss_jump = rss_now - last_rss
        is_step = rss_jump > step_threshold_mb
        
        data_case3.append({
            'batch_num': i,
            'rss_mb': rss_now,
            'dict_size': dict_size,
            'traced_mb': traced_mb,
            'rss_jump': rss_jump,
            'is_step': is_step
        })
        
        if is_step:
            print(f"STEP at Batch {i}: RSS jumped by {rss_jump:.2f} MB")
        elif i % (LOG_INTERVAL * 50) == 0:
            print(f"  Batch {i}/{TOTAL_BATCHES} - RSS: {rss_now:.2f} MB - Dict: {dict_size:,} items")
        
        last_rss = rss_now

print("\nCase 3 complete.\n")

# --- CASE 4: ADD WITH DUPLICATES THEN EVICT ---
print("="*70)
print(f"CASE 4: ADD STRINGS WITH {DUPLICATE_PERCENTAGE}% DUPLICATES THEN EVICT")
print("="*70)

gc.collect()
tracemalloc.stop()
tracemalloc.start()

global_word_cache_2 = {}
reusable_pool_2 = []
all_batches = []

# Phase 1: Add with duplicates
print("Phase 1: Adding strings with duplicates...")
last_rss = get_rss_mb()

for i in range(TOTAL_BATCHES):
    new_batch = []
    
    # Add unique strings
    for _ in range(NUM_UNIQUE):
        new_string = str(uuid.uuid4())
        new_batch.append(new_string)
        reusable_pool_2.append(new_string)
    
    # Add duplicate strings
    if len(reusable_pool_2) >= NUM_DUPLICATES:
        duplicates = random.sample(reusable_pool_2, NUM_DUPLICATES)
        new_batch.extend(duplicates)
    else:
        for _ in range(NUM_DUPLICATES):
            new_batch.append(str(uuid.uuid4()))
    
    all_batches.append(new_batch)
    
    for word in new_batch:
        if word not in global_word_cache_2:
            global_word_cache_2[word] = True
    
    if i % LOG_INTERVAL == 0:
        rss_now = get_rss_mb()
        dict_size = len(global_word_cache_2)
        current_traced, peak_traced = tracemalloc.get_traced_memory()
        traced_mb = current_traced / (1024 * 1024)
        
        rss_jump = rss_now - last_rss
        is_step = rss_jump > step_threshold_mb
        
        data_case4.append({
            'batch_num': i,
            'phase': 'add',
            'rss_mb': rss_now,
            'dict_size': dict_size,
            'traced_mb': traced_mb,
            'rss_jump': rss_jump,
            'is_step': is_step
        })
        
        if i % (LOG_INTERVAL * 50) == 0:
            print(f"  Add Batch {i}/{TOTAL_BATCHES} - RSS: {rss_now:.2f} MB - Dict: {dict_size:,}")
        
        last_rss = rss_now

print(f"\nPhase 1 complete. Peak RSS: {rss_now:.2f} MB, Dict size: {dict_size:,}\n")

# Phase 2: Evict
print("Phase 2: Evicting strings (batch by batch)...")
last_rss = rss_now

for i in range(TOTAL_BATCHES):
    batch_to_evict = all_batches[i]
    for word in batch_to_evict:
        if word in global_word_cache_2:
            del global_word_cache_2[word]
    
    if i % LOG_INTERVAL == 0:
        gc.collect()
        rss_now = get_rss_mb()
        dict_size = len(global_word_cache_2)
        current_traced, peak_traced = tracemalloc.get_traced_memory()
        traced_mb = current_traced / (1024 * 1024)
        
        rss_jump = rss_now - last_rss
        
        data_case4.append({
            'batch_num': TOTAL_BATCHES + i,
            'phase': 'evict',
            'rss_mb': rss_now,
            'dict_size': dict_size,
            'traced_mb': traced_mb,
            'rss_jump': rss_jump,
            'is_step': False
        })
        
        if i % (LOG_INTERVAL * 50) == 0:
            print(f"  Evict Batch {i}/{TOTAL_BATCHES} - RSS: {rss_now:.2f} MB - Dict: {dict_size:,}")
        
        last_rss = rss_now

print(f"\nPhase 2 complete. Final RSS: {rss_now:.2f} MB, Dict size: {dict_size}\n")

tracemalloc.stop()

# --- PLOTTING ---
print("="*70)
print("GENERATING PLOTS")
print("="*70)

df_case3 = pd.DataFrame(data_case3)
df_case4 = pd.DataFrame(data_case4)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Case 3: Plot 1 - RSS Memory
ax1 = axes[0, 0]
ax1.plot(df_case3['batch_num'], df_case3['rss_mb'], 'b-', linewidth=2, label='RSS Memory')
ax1.plot(df_case3['batch_num'], df_case3['traced_mb'], 'r--', linewidth=1.5, alpha=0.7, label='Traced Memory')
major_steps = df_case3[df_case3['is_step'] == True]
for idx, row in major_steps.iterrows():
    ax1.annotate(f"+{row['rss_jump']:.0f}MB", xy=(row['batch_num'], row['rss_mb']),
                xytext=(10, 10), textcoords='offset points', fontsize=8, color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax1.set_title(f'CASE 3: Memory (Add {DUPLICATE_PERCENTAGE}% Duplicates)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Batch Number')
ax1.set_ylabel('Memory (MB)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.4)

# Case 3: Plot 2 - Dictionary Size
ax2 = axes[0, 1]
ax2.plot(df_case3['batch_num'], df_case3['dict_size'], 'g-', linewidth=2, label='Dictionary Size')
ax2.set_title(f'CASE 3: Dictionary Growth ({DUPLICATE_PERCENTAGE}% Duplicates)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Batch Number')
ax2.set_ylabel('Items')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.4)

# Case 4: Plot 1 - RSS Memory (Add + Evict)
ax3 = axes[1, 0]
df_add = df_case4[df_case4['phase'] == 'add']
df_evict = df_case4[df_case4['phase'] == 'evict']
ax3.plot(df_add['batch_num'], df_add['rss_mb'], 'b-', linewidth=2, label='RSS (Add Phase)')
ax3.plot(df_evict['batch_num'], df_evict['rss_mb'], 'purple', linewidth=2, label='RSS (Evict Phase)')
ax3.axvline(x=TOTAL_BATCHES, color='red', linestyle='--', linewidth=2, label='Eviction Starts')
ax3.set_title(f'CASE 4: Memory (Add {DUPLICATE_PERCENTAGE}% Dup Then Evict)', fontsize=13, fontweight='bold')
ax3.set_xlabel('Batch Number')
ax3.set_ylabel('Memory (MB)')
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.4)

# Case 4: Plot 2 - Dictionary Size (Add + Evict)
ax4 = axes[1, 1]
ax4.plot(df_add['batch_num'], df_add['dict_size'], 'g-', linewidth=2, label='Dict Size (Add)')
ax4.plot(df_evict['batch_num'], df_evict['dict_size'], 'orange', linewidth=2, label='Dict Size (Evict)')
ax4.axvline(x=TOTAL_BATCHES, color='red', linestyle='--', linewidth=2, label='Eviction Starts')
ax4.set_title(f'CASE 4: Dictionary Growth ({DUPLICATE_PERCENTAGE}% Dup, Add Then Evict)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Batch Number')
ax4.set_ylabel('Items')
ax4.legend()
ax4.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_FILENAME, dpi=150)
print(f"✅ Graph saved as '{OUTPUT_FILENAME}'\n")

# Save data
df_case3.to_csv(f'case3_duplicate_{DUPLICATE_PERCENTAGE}pct_add_only.csv', index=False)
df_case4.to_csv(f'case4_duplicate_{DUPLICATE_PERCENTAGE}pct_add_evict.csv', index=False)
print("✅ Data saved to CSV files\n")

plt.show()
