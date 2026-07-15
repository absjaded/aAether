import numpy as np
from pathlib import Path

def inspect_npy(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"File not found: {p}")
        return
    data = np.load(p)
    print(f"--- {p.name} ---")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}")
    if "mask" in p.name:
        print(f"Unique values: {np.unique(data)}")
    print()

import numpy as np
from pathlib import Path

def inspect_npy(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"File not found: {p}")
        return
    data = np.load(p)
    print(f"--- {p.name} ---")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}")
    if "mask" in p.name:
        print(f"Unique values: {np.unique(data)}")
    print()

def main():
    base_dir = str(Path(__file__).resolve().parent.parent / "split_data")
    
    # 1. Tribe features
    inspect_npy(f"{base_dir}/tribe/scene1_tribe_cortical.npy")
    
    # 2. Omni latents
    inspect_npy(f"{base_dir}/omni/whole_brain/scene1_latents.npy")
    inspect_npy(f"{base_dir}/omni/whole_brain/scene1_mask.npy")
    inspect_npy(f"{base_dir}/omni/whole_brain/scene1_coords.npy")
    inspect_npy(f"{base_dir}/omni/whole_brain/scene1_cls.npy")

if __name__ == "__main__":
    main()
