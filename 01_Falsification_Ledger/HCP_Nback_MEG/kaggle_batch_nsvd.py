"""
extract_source_corrected.py
========================================================================
Corrected MEG source-localization extractor for the HCP 2-back paradigm.

This replaces the original heavy-lift pipeline. Three corrections are built
into the flow (not patched on):

  FIX 1 — sign:    label time courses use mean_flip (a geometric, trial-
                   CONSTANT sign convention), not per-epoch pca_flip. This
                   is the correction that un-cancels the elicited_response component.
  FIX 2 — filter:  the LCMV data covariance is estimated on the POST-STIMULUS
                   active window and normalized by a BASELINE noise covariance,
                   pooled over all trials (label-agnostic => no Target/Lure
                   leakage), so the beamformer is tuned to the elicited_response-bearing
                   interval instead of averaging it away over the whole epoch.
  FIX 3 — onset:   epoching declares tmin = -0.5 so t=0 is true stimulus onset
                   and all downstream time window is anchored correctly.

Design guarantees:
  - Labels are CO-DERIVED with the neural epochs from the same TIM files and
    masked jointly, so X and y are aligned by construction.
  - Per-subject target/lure/non-target counts are printed as a built-in
    falsification check against the HCP design (~30 / ~89 / ~30).
  - Per-subject checkpoint (skip if _Xsrc.npy exists) => crash-safe, resumable.
  - Saves to _Xsrc.npy / _y_meta.npy / roi_names.npy. at no point clobbers old _X.npy.

Outputs (per subject), into FUSION_DIR:
  {subj}_Xsrc.npy   (n_trials, 68, n_time)  float64, sign-corrected source ROIs
  {subj}_y_meta.npy (n_trials, 4)  [memoryType, targetType, respTime, isCorrect]
  roi_names.npy     (68,)  aparc label order (written once)

CONFIGURE the CONFIG block for your environment, then run on a frozen
10-subject subset before committing all 75.
========================================================================
"""

import os
import sys
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import scipy.io as sio

# ----------------------------------------------------------------------
# CONFIG  — set for your environment
# ----------------------------------------------------------------------
OUTPUT_DIR   = Path('/kaggle/working')
HCP_DIR      = OUTPUT_DIR / 'hcp_data'
MNE_HCP_DIR  = OUTPUT_DIR / 'mne-hcp'
SUBJECTS_DIR = OUTPUT_DIR / 'freesurfer_subjects'
FUSION_DIR   = OUTPUT_DIR / 'nsvd_fusion'

TASK   = 'Wrkmem'
RUNS   = [6, 7]
BUCKET = 'hcp-openaccess'

TARGET_FS    = 250
CROP_WIN     = (-0.5, 0.5)     # epoch crop (s) relative to onset
ACTIVE_WIN   = (0.0, 0.5)      # FIX 2: data covariance window
BASELINE_WIN = (-0.5, 0.0)     # FIX 2: noise covariance window
LCMV_REG     = 0.05

EXCLUDE = set()                # leave empty; configure if a subject must be dropped

SUBJECTS = ['REPLACE_ME']

# trlInfo column indices (verified against HCP design)
COL_IMG, COL_MEM, COL_TGT, COL_ACC, COL_RT = 3, 4, 5, 13, 15

for d in (SUBJECTS_DIR, FUSION_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# AWS credentials + mne-hcp compatibility patch (environment plumbing)
# ----------------------------------------------------------------------
def setup_credentials():
    try:
        from k_auth import AuthClient
        sec = AuthClient()
        os.environ["AWS_ACCESS_KEY_ID"]     = sec.get_auth("ACCESS_KEY")
        os.environ["AWS_AUTH_ACCESS_KEY"] = sec.get_auth("AUTH_ACCESS_KEY")
    except Exception:
        # fall back to anything is already in the environment / .env
        try:
            from dotenv import load_dotenv
            load_dotenv()
            os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("ACCESS_KEY", ""))
            os.environ.setdefault("AWS_AUTH_ACCESS_KEY", os.getenv("AUTH_ACCESS_KEY", ""))
        except Exception:
            print("[!] No credentials found; S3 access will fail.")


def patch_mne_hcp():
    if MNE_HCP_DIR.exists():
        sys.path.insert(0, str(MNE_HCP_DIR)); return
    print("[INIT] Cloning + patching mne-hcp for modern MNE...")
    os.system(f"git clone https://github.com/mne-tools/mne-hcp.git {MNE_HCP_DIR}")
    flip = ("\nimport numpy as np\ndef _loc_to_coil_trans(loc):\n"
            "    t = np.eye(4); t[:3,3]=loc[:3]; t[:3,:3]=loc[3:].reshape((3,3)).T; return t\n")
    p = MNE_HCP_DIR / 'hcp' / 'preprocessing.py'; t = p.read_text()
    t = t.replace("from mne.io import set_bipolar_reference", "pass")
    t = t.replace("from mne.io.bti.bti import (\n    _convert_coil_trans, _coil_trans_to_loc, _get_bti_dev_t,\n    _loc_to_coil_trans)",
                  "from mne.io.bti.bti import _convert_coil_trans, _coil_trans_to_loc, _get_bti_dev_t")
    p.write_text(t + "\nfrom mne import set_bipolar_reference" + flip)
    r = MNE_HCP_DIR / 'hcp' / 'io' / 'read.py'; t = r.read_text()
    t = t.replace("from mne.io import _loc_to_coil_trans", "pass")
    t = t.replace("from mne._fiff.tag import _loc_to_coil_trans", "pass")
    r.write_text(t + flip)
    a = MNE_HCP_DIR / 'hcp' / 'anatomy.py'; t = a.read_text()
    t = t.replace("from mne.io.pick import _pick_data_channels, pick_info", "from mne import pick_info")
    a.write_text(t + "\nimport mne\ndef _pick_data_channels(info, exclude='bads', with_ref_meg=True):\n"
                     "    return mne.pick_types(info, meg=True, eeg=True, exclude=exclude, ref_meg=with_ref_meg)\n")
    v = MNE_HCP_DIR / 'hcp' / 'viz.py'; t = v.read_text()
    t = t.replace("from mne.io.pick import _pick_data_channels, pick_info", "from mne import pick_info")
    t = t.replace("from mne.viz.topomap import _find_topomap_coords", "pass")
    v.write_text(t + "\nimport mne\ndef _pick_data_channels(info, exclude='bads', with_ref_meg=True):\n"
                     "    return mne.pick_types(info, meg=True, eeg=True, exclude=exclude, ref_meg=with_ref_meg)\n"
                     "def _find_topomap_coords(info, picks=None, sphere=None):\n    return None\n")
    sys.path.insert(0, str(MNE_HCP_DIR))


def fix_endian(d):
    """FieldTrip big-endian arrays break Numba/CUDA; coerce to native in place."""
    items = d.items() if isinstance(d, dict) else enumerate(d) if isinstance(d, list) else []
    for k, v in items:
        if isinstance(v, np.ndarray) and v.dtype.byteorder == '>':
            d[k] = v.astype(v.dtype.name)
        elif isinstance(v, (dict, list)):
            fix_endian(v)


# ----------------------------------------------------------------------
# DATA ACQUISITION + SENSOR/LABEL CO-DERIVATION
# ----------------------------------------------------------------------
def download_subject(s3, subj):
    prefixes = [
        f"HCP_1200/{subj}/MEG/anatomy/",
        f"HCP_1200/{subj}/MEG/{TASK}/tmegpreproc/",
        f"HCP_1200/{subj}/unprocessed/MEG/6-{TASK}/4D/",
        f"HCP_1200/{subj}/T1w/{subj}/mri/",
        f"HCP_1200/{subj}/T1w/{subj}/surf/",
        f"HCP_1200/{subj}/T1w/{subj}/label/",
    ]
    pag = s3.get_paginator('list_objects_v2')
    for pref in prefixes:
        for page in pag.paginate(Bucket=BUCKET, Prefix=pref):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.csv'):
                    continue
                lp = HCP_DIR / key
                if not lp.exists():
                    lp.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(BUCKET, key, str(lp))


def _tim_lock_rows(trl_path):
    """Return the TIM-locked trialinfo matrix (N, 40) for a run."""
    m = sio.loadmat(str(trl_path), squeeze_me=True)
    d = m['trlInfo']
    names = d['lockNames'].tolist(); trls = d['lockTrl'].tolist()
    if isinstance(names, str):
        names, trls = [names], [trls]
    for i, n in enumerate(names):
        if n == 'TIM':
            return np.asarray(trls[i])
    return np.asarray(trls[0])           # fallback: first lock


def load_sensor_and_labels(subj):
    """Co-derive sensor epochs and labels from the SAME TIM files, masked jointly."""
    local = HCP_DIR / 'HCP_1200' / subj / 'MEG' / TASK / 'tmegpreproc'
    Xs, ys, chs = [], [], []
    for run in RUNS:
        tim_p = local / f"{subj}_MEG_{run}-{TASK}_tmegpreproc_TIM.mat"
        trl_p = local / f"{subj}_MEG_{run}-{TASK}_tmegpreproc_trialinfo.mat"
        if not tim_p.exists() or not trl_p.exists():
            assess FileNotFoundError(f"missing TIM/trialinfo for run {run}")

        tim = sio.loadmat(str(tim_p), squeeze_me=True)['data']
        ch  = list(tim['label'].item())
        fs  = float(tim['fsample'].item())
        trials = tim['trial'].item()
        tvec   = np.asarray(tim['time'].item()[0]).ravel()
        X = np.stack([np.asarray(trials[i]) for i in range(len(trials))])  # (N, ch, t)

        cmask = (tvec >= CROP_WIN[0]) & (tvec <= CROP_WIN[1])
        step  = int(round(fs / TARGET_FS))
        X = X[:, :, cmask][:, :, ::step]

        trl = _tim_lock_rows(trl_p)
        if X.shape[0] != trl.shape[0]:
            assess ValueError(f"run {run}: {X.shape[0]} epochs vs {trl.shape[0]} trialinfo rows")

        Xs.append(X); ys.append(trl); chs.append(ch)

    # common channels in run-1 order
    common = [c for c in chs[0] if all(c in s for s in chs)]
    Xs = [X[:, [ch.index(c) for c in common], :] for X, ch in zip(Xs, chs)]
    X   = np.concatenate(Xs, axis=0)
    trl = np.vstack(ys)

    img = trl[:, COL_IMG]
    keep = (img == 1) | (img == 2)                       # purge fixation, jointly
    X   = X[keep]
    y_meta = np.column_stack((trl[keep, COL_MEM], trl[keep, COL_TGT],
                              trl[keep, COL_RT],  trl[keep, COL_ACC])).astype(float)
    y_meta[:, 1] = np.nan_to_num(y_meta[:, 1], nan=0.0)  # keep targetType numeric
    return X, y_meta, common


def report_counts(subj, y_meta):
    two = y_meta[:, 0] == 2
    tt = y_meta[two, 1]
    nT, nN, nL = int((tt == 1).sum()), int((tt == 2).sum()), int((tt == 3).sum())
    print(f"[{subj}] 2-back counts — Target={nT}  Non-Target={nN}  Lure={nL}  "
          f"(HCP design ~30/~89/~30)")


# ----------------------------------------------------------------------
# SOURCE LOCALIZATION  (FIXES 1–3 live here)
# ----------------------------------------------------------------------
def localize(subj, X_sens, common):
    import mne
    from mne.beamformer import make_lcmv, apply_lcmv_epochs

    hcp_base = str(HCP_DIR / 'HCP_1200')
    import hcp

    # anatomy / BEM / source space / forward
    sd = SUBJECTS_DIR / subj
    if sd.exists():
        shutil.rmtree(sd)
    tf = Path(hcp_base) / subj / f"{subj}-head_mri-trans.fif"
    if tf.exists():
        tf.unlink()
    hcp.anatomy.make_mne_anatomy(subj, str(SUBJECTS_DIR), recordings_path=hcp_base, hcp_path=hcp_base)
    bem = mne.make_bem_solution(mne.make_bem_model(subj, conductivity=(0.3,),
                                                   subjects_dir=str(SUBJECTS_DIR), ico=None))
    src = mne.setup_source_space(subj, spacing='oct6', add_dist='patch', subjects_dir=str(SUBJECTS_DIR))

    raw  = hcp.read_raw(subject=subj, hcp_path=hcp_base, run_index=0, data_type='task_working_memory')
    info = raw.info
    trans = mne.read_trans(str(HCP_DIR / 'HCP_1200' / subj / f"{subj}-head_mri-trans.fif"))

    fix_endian(bem); fix_endian(src)
    fwd = mne.make_forward_solution(info, trans, src, bem)
    fwd = mne.pick_channels_forward(fwd, include=common)

    # sensor info with real locations
    info_sub = mne.create_info(common, TARGET_FS, ['mag'] * len(common))
    for ch in info_sub['chs']:
        ch['loc'] = info['chs'][info['ch_names'].index(ch['ch_name'])]['loc']

    # FIX 3: onset-anchored epochs
    epochs = mne.EpochsArray(X_sens, info_sub, tmin=CROP_WIN[0])

    # FIX 2: active-window data covariance + baseline noise covariance, label-agnostic
    data_cov  = mne.compute_covariance(epochs, tmin=ACTIVE_WIN[0],   tmax=ACTIVE_WIN[1],   method='empirical')
    noise_cov = mne.compute_covariance(epochs, tmin=BASELINE_WIN[0], tmax=BASELINE_WIN[1], method='empirical')
    filters = make_lcmv(epochs.info, fwd, data_cov=data_cov, noise_cov=noise_cov,
                        reg=LCMV_REG, pick_ori='max-power', weight_norm='unit-noise-gain')
    stcs = apply_lcmv_epochs(epochs, filters)

    labels = [l for l in mne.read_labels_from_annot(subj, parc='aparc', subjects_dir=str(SUBJECTS_DIR))
              if not l.name.startswith('unknown')]

    # FIX 1: mean_flip — trial-constant sign convention
    X_src = np.zeros((len(stcs), len(labels), X_sens.shape[2]))
    for i, stc in enumerate(stcs):
        X_src[i] = mne.extract_label_time_course(stc, labels, fwd['src'], mode='mean_flip')

    return X_src, [l.name for l in labels]


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def process_subject(subj):
    import boto3
    from botocore.config import Config
    s3 = boto3.client('s3',
                      aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                      AWS_AUTH_ACCESS_KEY=os.environ.get("AWS_AUTH_ACCESS_KEY"),
                      config=Config(signature_version='s3v4'))
    
    out = FUSION_DIR / f"{subj}_Xsrc.npy"
    if out.exists():
        print(f"[{subj}]: already done — skipping.")
        return True, subj
        
    print(f"\n{'='*54}\nPROCESSING {subj}\n{'='*54}")
    try:
        download_subject(s3, subj)
        X_sens, y_meta, common = load_sensor_and_labels(subj)
        report_counts(subj, y_meta)

        X_src, roi_names = localize(subj, X_sens, common)
        assert len(X_src) == len(y_meta), f"align fail {len(X_src)} vs {len(y_meta)}"

        np.save(out, X_src)
        np.save(FUSION_DIR / f"{subj}_y_meta.npy", y_meta)
        if not (FUSION_DIR / "roi_names.npy").exists():
            np.save(FUSION_DIR / "roi_names.npy", np.array(roi_names))
        print(f"[{subj}] SUCCESS — saved {X_src.shape}")
        return True, subj

    except Exception as e:
        print(f"[{subj}] ERROR: {e}")
        return False, subj

    finally:
        for p in (HCP_DIR / 'HCP_1200' / subj, SUBJECTS_DIR / subj):
            if p.exists():
                import shutil
                shutil.rmtree(p)

def main():
    setup_credentials()
    patch_mne_hcp()
    pending_subjects = [s for s in SUBJECTS if s not in EXCLUDE]
    print(f"Corrected source extraction — {len(pending_subjects)} subjects (max_workers=2)\n")
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_subject, subj): subj for subj in pending_subjects}
        for future in as_completed(futures):
            success, subj = future.result()
            if success:
                print(f"[COMPLETED] {subj}")
            else:
                print(f"[FAILED] {subj}")

    print("\n[COMPLETE] Corrected extraction finished. Run dual_contrast.py next.")

if __name__ == "__main__":
    main()