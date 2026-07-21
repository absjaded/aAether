import sys
import os
import torch
# LaBraM is in the workspace
sys.path.insert(0, r"../LaBraM")
try:
    import modeling_pretrain
    from timm.models import create_model
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
# Ensure checkpoint is available
import huggingface_hub
print("Downloading checkpoint...")
ckpt_path = huggingface_hub.hub_download(repo_id="braindecode/labram-pretrained", filename="pytorch_model.bin")
print("Creating model...")
labram_model = create_model("labram_base_patch200_200", pretrained=False, num_classes=0, init_values=0.1)
print("Loading checkpoint...")
checkpoint = torch.load(ckpt_path, map_location="cpu")
ckpt_dict = checkpoint.get("model", checkpoint)
print("\nModel Keys (first 10):")
print(list(labram_model.state_dict().keys())[:10])
print("\nCheckpoint Keys (first 10):")
print(list(ckpt_dict.keys())[:10])
print("\nKeys ending in weight in model (first 5):")
print([k for k in labram_model.state_dict().keys() if 'weight' in k][:5])
print("\nKeys ending in weight in checkpoint (first 5):")
print([k for k in ckpt_dict.keys() if 'weight' in k][:5])
