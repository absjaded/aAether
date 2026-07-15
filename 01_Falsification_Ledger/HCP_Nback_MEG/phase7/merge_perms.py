import numpy as np
import glob
import os

def main():
    print("Looking for null_shard_*.npy files in the current directory...")
    
    # Find all shard files
    shard_files = sorted(glob.glob("null_shard_*.npy"))
    
    if not shard_files:
        print("Error: No 'null_shard_*.npy' files found in the current directory.")
        print("Please download the output files from your 6 Kaggle recordbooks and place them here.")
        return

    print(f"Found {len(shard_files)} shards: {shard_files}")
    
    # Load and concatenate all permutations
    all_nulls = []
    for f in shard_files:
        shard_data = np.load(f)
        all_nulls.append(shard_data)
        
    final_null_distribution = np.concatenate(all_nulls)
    n_perms = len(final_null_distribution)
    print(f"\\nSuccessfully merged into a final distribution of {n_perms} permutations!")
    
    # Try to load the observed gaps
    if os.path.exists("mvp0_gaps.npy"):
        gaps = np.load("mvp0_gaps.npy")
        observed = gaps.mean()
        
        # Calculate p-value: proportion of null permutations whose absolute mean 
        # is greater than or equal to the absolute observed mean
        perm_p = float(np.mean(np.abs(final_null_distribution) >= np.abs(observed)))
        
        print(f"\\n=== FINAL PERMUTATION RESULTS ===")
        print(f"Observed d        : {observed:+.4f}")
        print(f"Null mean +/- SD  : {final_null_distribution.mean():+.4f} +/- {final_null_distribution.std():.4f}")
        print(f"Permutation p     : {perm_p:.4f}")
        print("=================================")
    else:
        print("\\nrecord: 'mvp0_gaps.npy' not found in current directory.")
        print("Make sure to download it from one of the Kaggle outputs if you want the p-value calculated automatically.")

if __name__ == '__main__':
    main()
