import os
import psutil
import gc
import uuid
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
OUTPUT_FILENAME = 'case1_case2_unique_strings.png'

# --- DATA COLLECTION ---
data_case1 = []  # Add only
data_case2 = []  # Add then evict

# --- START ---
print("="*70)
print("CASE 1: ADD UNIQUE STRINGS")
print("="*70)
gc.collect()
baseline_rss = get_rss_mb()
print(f"Baseline RSS: {baseline_rss:.2f} MB\n")

tracemalloc.start()

# CASE 1: Add unique strings
global_word_cache = {}
last_rss = baseline_rss
step_threshold_mb = 50

for i in range(TOTAL_BATCHES):
    # Create new unique strings
    new_batch = [str(uuid.uuid4()) for _ in range(BATCH_SIZE)]
    
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
        
        data_case1.append({
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

print("\nCase 1 complete.\n")

# --- CASE 2: ADD THEN EVICT ---
print("="*70)
print("CASE 2: ADD UNIQUE STRINGS THEN EVICT (Memory Zone Simulation)")
print("="*70)

gc.collect()
tracemalloc.stop()
tracemalloc.start()

global_word_cache_2 = {}
all_batches = []  # Store batches for later eviction

# Phase 1: Add (same as Case 1)
print("Phase 1: Adding strings...")
last_rss = get_rss_mb()

for i in range(TOTAL_BATCHES):
    new_batch = [str(uuid.uuid4()) for _ in range(BATCH_SIZE)]
    all_batches.append(new_batch)  # Store for eviction
    
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
        
        data_case2.append({
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

# Phase 2: Evict (Memory Zone cleanup)
print("Phase 2: Evicting strings (batch by batch)...")
last_rss = rss_now

for i in range(TOTAL_BATCHES):
    # Evict the batch
    batch_to_evict = all_batches[i]
    for word in batch_to_evict:
        if word in global_word_cache_2:
            del global_word_cache_2[word]
    
    if i % LOG_INTERVAL == 0:
        gc.collect()  # Force garbage collection
        rss_now = get_rss_mb()
        dict_size = len(global_word_cache_2)
        current_traced, peak_traced = tracemalloc.get_traced_memory()
        traced_mb = current_traced / (1024 * 1024)
        
        rss_jump = rss_now - last_rss
        
        data_case2.append({
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

df_case1 = pd.DataFrame(data_case1)
df_case2 = pd.DataFrame(data_case2)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Case 1: Plot 1 - RSS Memory
ax1 = axes[0, 0]
ax1.plot(df_case1['batch_num'], df_case1['rss_mb'], 'b-', linewidth=2, label='RSS Memory')
ax1.plot(df_case1['batch_num'], df_case1['traced_mb'], 'r--', linewidth=1.5, alpha=0.7, label='Traced Memory')
major_steps = df_case1[df_case1['is_step'] == True]
for idx, row in major_steps.iterrows():
    ax1.annotate(f"+{row['rss_jump']:.0f}MB", xy=(row['batch_num'], row['rss_mb']),
                xytext=(10, 10), textcoords='offset points', fontsize=8, color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax1.set_title('CASE 1: Memory (Add Unique Strings)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Batch Number')
ax1.set_ylabel('Memory (MB)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.4)

# Case 1: Plot 2 - Dictionary Size
ax2 = axes[0, 1]
ax2.plot(df_case1['batch_num'], df_case1['dict_size'], 'g-', linewidth=2, label='Dictionary Size')
ax2.set_title('CASE 1: Dictionary Growth', fontsize=13, fontweight='bold')
ax2.set_xlabel('Batch Number')
ax2.set_ylabel('Items')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.4)

# Case 2: Plot 1 - RSS Memory (Add + Evict)
ax3 = axes[1, 0]
df_add = df_case2[df_case2['phase'] == 'add']
df_evict = df_case2[df_case2['phase'] == 'evict']
ax3.plot(df_add['batch_num'], df_add['rss_mb'], 'b-', linewidth=2, label='RSS (Add Phase)')
ax3.plot(df_evict['batch_num'], df_evict['rss_mb'], 'purple', linewidth=2, label='RSS (Evict Phase)')
ax3.axvline(x=TOTAL_BATCHES, color='red', linestyle='--', linewidth=2, label='Eviction Starts')
ax3.set_title('CASE 2: Memory (Add Then Evict)', fontsize=13, fontweight='bold')
ax3.set_xlabel('Batch Number')
ax3.set_ylabel('Memory (MB)')
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.4)

# Case 2: Plot 2 - Dictionary Size (Add + Evict)
ax4 = axes[1, 1]
ax4.plot(df_add['batch_num'], df_add['dict_size'], 'g-', linewidth=2, label='Dict Size (Add)')
ax4.plot(df_evict['batch_num'], df_evict['dict_size'], 'orange', linewidth=2, label='Dict Size (Evict)')
ax4.axvline(x=TOTAL_BATCHES, color='red', linestyle='--', linewidth=2, label='Eviction Starts')
ax4.set_title('CASE 2: Dictionary Growth (Add Then Evict)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Batch Number')
ax4.set_ylabel('Items')
ax4.legend()
ax4.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_FILENAME, dpi=150)
print(f"✅ Graph saved as '{OUTPUT_FILENAME}'\n")

# Save data
df_case1.to_csv('case1_unique_add_only.csv', index=False)
df_case2.to_csv('case2_unique_add_evict.csv', index=False)
print("✅ Data saved to CSV files\n")

plt.show()
