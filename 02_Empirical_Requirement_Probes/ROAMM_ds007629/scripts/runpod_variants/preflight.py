#!/usr/bin/env python3
"""
preflight.py — run this FIRST on the pod. It takes ~60s and fails loudly.
Every check here corresponds to a failure that would otherwise surface hours in.

  python preflight.py
"""
import os, sys, io, json, pickle, urllib.request, warnings
import numpy as np
warnings.filterwarnings('ignore')

FAIL = []
def check(name, fn):
    try:
        r = fn()
        print(f"  [OK]   {name}" + (f"  -> {r}" if r else ""))
    except Exception as e:
        print(f"  [FAIL] {name}\n         {type(e).__name__}: {str(e)[:200]}")
        FAIL.append(name)

ROAMM_CH = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7',
 'CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz',
 'Fpz','Fp2','AF8','AF4','Afz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz',
 'C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']
RENAME = {'Afz': 'AFz'}                      # the ONE name the position bank rejects
CH = [RENAME.get(c, c) for c in ROAMM_CH]

print("\n=== 1. HF auth + gated weights ===")
def _tok():
    t = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not t:
        from huggingface_hub import HfFolder
        t = HfFolder.get_token()
    assert t, "No HF token. export HF_TOKEN=... or run: huggingface-cli login"
    return "token found"
check("HF token present", _tok)

def _gated():
    from huggingface_hub import HfApi
    HfApi().model_info("brain-bzh/reve-base")   # raises if not accepted
    return "reve-base accessible"
check("reve-base gate accepted", _gated)

print("\n=== 2. Position bank resolves ALL 64 channels ===")
def _pos():
    from huggingface_hub import hf_hub_download
    bank = json.load(open(hf_hub_download("brain-bzh/reve-positions", "positions.json")))
    if isinstance(bank, dict) and 'positions' in bank: bank = bank['positions']
    miss = [c for c in CH if c not in bank]
    assert not miss, f"UNRESOLVED CHANNELS: {miss}"
    return f"all {len(CH)}/64 resolve (after renaming {RENAME})"
check("64/64 channel names in position bank", _pos)

print("\n=== 3. REVE loads and does a forward pass ===")
def _reve():
    import torch
    from braindecode.models import REVE
    m = REVE.from_pretrained("brain-bzh/reve-base",
                             n_chans=64, n_times=400, sfreq=200,
                             chs_info=[{"ch_name": c} for c in CH], n_outputs=2)
    m.eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = m.to(dev)
    pos = m.get_positions(CH).to(dev)                 # (64,3)
    x = torch.randn(2, 64, 400, device=dev)           # 2s @ 200Hz
    with torch.no_grad():
        out = m(x, pos=pos, return_features=True)
    f = out["features"]
    return f"device={dev}  features={tuple(f.shape)}  (batch, tokens/dims)"
check("REVE forward pass, return_features", _reve)

print("\n=== 4. ROAMM data reachable + EEG present (non-10014 subject) ===")
def _data():
    url = ("https://s3.amazonaws.com/openneuro.org/ds007629/derivatives/synced/"
           "sub-10052/sub-10052_task-ReMind_run-01_mldata.pkl")
    buf = io.BytesIO(urllib.request.urlopen(url, timeout=120).read())   # RAM, not disk
    df = pickle.load(buf)
    have = [c for c in ROAMM_CH if c in df.columns]
    assert len(have) == 64, f"only {len(have)}/64 EEG channels"
    assert 'is_mw' in df.columns, "no is_mw label"
    sf = float(df['sfreq'].dropna().iloc[0])
    mx = float(np.nanmax(np.abs(df[have[:8]].values)))
    unit = "VOLTS (multiply by 1e6!)" if mx < 1e-2 else "microvolts"
    del df, buf
    return f"64/64 EEG ch, is_mw present, sfreq={sf}Hz, units={unit}"
check("synced pickle streams from S3 with EEG", _data)

print("\n=== 5. GPU + RAM headroom ===")
def _hw():
    import torch, psutil
    ram = psutil.virtual_memory().total / 1e9
    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        vram = g.total_memory / 1e9
        assert vram > 20, f"only {vram:.0f}GB VRAM"
        return f"{g.name} {vram:.0f}GB VRAM | {ram:.0f}GB RAM | disk-free {os.statvfs('/').f_bavail*os.statvfs('/').f_frsize/1e9:.0f}GB"
    return f"NO GPU | {ram:.0f}GB RAM"
check("hardware", _hw)

print("\n" + "="*60)
if FAIL:
    print(f"PREFLIGHT FAILED: {FAIL}")
    print("Fix these before starting the extraction. Do NOT run the long job.")
    sys.exit(1)
print("PREFLIGHT PASSED — safe to start the extraction.")
print("""
REMINDERS BAKED IN FROM THE AUDIT:
  * rename Afz -> AFz  (the only channel the position bank rejects)
  * EEG is in VOLTS. Multiply by 1e6 before anything touches a covariance.
  * resample 256 -> 200 Hz (REVE requires 200 Hz).
  * stream pickles through RAM (io.BytesIO), never to disk. You have 30GB disk.
  * write embeddings PER SUBJECT to the network volume as you go.
  * embed the BALANCED subset only (~13.8k epochs -> ~5.4GB).
    All 176k epochs at full token grid = ~69GB. Do not.
""")
print("="*60)
