import os
import psutil
import gc
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import tracemalloc

# --- CONFIGURATION ---
TOTAL_BATCHES = 5000
BATCH_SIZE = 600  
LOG_INTERVAL = 50
OUTPUT_FILENAME = 'case2_with_zone_FIXED.png'

def get_rss_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# --- START ---
print("="*60)
print(f"SCRIPT 2: WITH MEMORY ZONE (Simulating Fix)")
print(f"NOTE: Any memory growth here is OS FRAGMENTATION, not Dict growth.")
print("="*60)

gc.collect()
baseline_rss = get_rss_mb()
print(f"Baseline RSS: {baseline_rss:.2f} MB\n")

tracemalloc.start()

# The "StringStore"
global_word_cache = {}
data_log = []
last_rss = baseline_rss

for i in range(TOTAL_BATCHES):
    # 1. Generate Batch of UNIQUE strings
    # IMPORTANT: We create this list LOCALLY inside the loop.
    # When the loop finishes, Python deletes this list.
    new_batch = [str(uuid.uuid4()) for _ in range(BATCH_SIZE)]
    
    # 2. ZONE ENTER: Add to Store
    for word in new_batch:
        if word not in global_word_cache:
            global_word_cache[word] = True
            
    # 3. ZONE EXIT: Evict immediately
    # This simulates the memory_zone exiting and pruning the new strings
    for word in new_batch:
        if word in global_word_cache:
            del global_word_cache[word]
            
    # 4. FORCE CLEANUP (To prove logic)
    # We ensure the 'new_batch' list itself is destroyed by Python
    # so it doesn't confuse our memory readings.
    if i % LOG_INTERVAL == 0:
        gc.collect()
        
        rss_now = get_rss_mb()
        dict_size = len(global_word_cache)
        rss_jump = rss_now - last_rss
        
        data_log.append({
            'batch': i,
            'rss': rss_now,
            'dict_size': dict_size,
            'jump': rss_jump
        })
        
        # Print ONLY if memory jumps significantly (Fragmentation Step)
        if rss_jump > 20: 
             print(f"⚠️ FRAGMENTATION STEP at Batch {i}: +{rss_jump:.1f} MB (Dict is still {dict_size})")
        elif i % (LOG_INTERVAL * 10) == 0:
            print(f"Batch {i} | RSS: {rss_now:.1f} MB | Dict: {dict_size} (Clean)")
            
        last_rss = rss_now

tracemalloc.stop()
print(f"\nDONE. Final Dict Size: {len(global_word_cache)}")

# --- PLOTTING ---
df = pd.DataFrame(data_log)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: RSS
ax1.plot(df['batch'], df['rss'], 'b-', label='RSS Memory (MB)')
ax1.set_title("CASE 2: Memory Usage (With Zone)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Memory (MB)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Correlation
ax2_twin = ax2.twinx()
ax2.plot(df['batch'], df['dict_size'], 'g-', linewidth=2, label='Dict Size (Should be 0)')
ax2_twin.plot(df['batch'], df['rss'], 'b--', alpha=0.3, label='RSS Trend')

ax2.set_title("Dictionary Growth vs Memory", fontsize=14, fontweight='bold')
ax2.set_xlabel("Batch Number")
ax2.set_ylabel("Items in Dictionary", color='green')
ax2_twin.set_ylabel("RSS Memory (MB)", color='blue')
# Set fixed scale for green line to emphasize it is 0
ax2.set_ylim(-10, 100) 
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig(OUTPUT_FILENAME)
print(f"Graph saved as {OUTPUT_FILENAME}")
plt.show()
