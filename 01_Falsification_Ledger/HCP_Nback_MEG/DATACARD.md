# Dataset Card: HCP MEG Working Memory N-back Task

## Identity
- **Dataset:** Human Connectome Project (HCP) MEG Working Memory Task
- **Modality:** Source-localized MEG — neural current (nAm), sign-corrected via mean_flip
- **Space:** 68 Desikan-Killiany cortical ROIs, 250 Hz, epochs −0.5s to +0.5s (255 samples, t=0 at sample index 127)
- **Task:** 2-back and 0-back working memory blocks. Analysis restricted to 2-back (load-matched) only.
- **Subjects analyzed:** Up to 75 subjects (admissibility-filtered to 45 for primary contrast)
- **Local data files:** `legacy/nsvd_fusion/{subj}_Xsrc.npy` and `{subj}_y_meta.npy` — **DO NOT COMMIT** (gitignored)
- **Distribution:** HCP data requires a signed Data Use Agreement. See https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms. **Raw .npy files must at no point be committed or shared publicly.**

## What Was Attempted
This dataset received the most extensive treatment of any dataset in this search. Across chats 13–23:
1. **v1.0 EDA audit** — basic loading, trial counts, label distribution
2. **v2.0 baseline** — full-trial SPD covariance, Tangent Space LR → Face/Tool semantic baseline (59.8% LOSO)
3. **v2.1–2.5 sweep series** — sliding window sweep, Riemannian distance sweep, proxy Riemannian with Euclidean Alignment, gate1 (load contrast), gate2 (friction/memory trace), gate3 (shuffle)
4. **Ordinal spacing test** — Lure vs. Non-Target vs. Target geodesic distance comparison (the definitive test)
5. **Lag regression** — continuous `distance ~ lag` OLS regression on withhold-only trials
6. **Motor negative control** — geodesic distance in motor ROIs (precentral, postcentral, paracentral) vs. primary ROIs
7. **Permutation testing** — 1000-sample label permutation null distribution

## The Falsification Result
**Ordinal spacing test:** Lure mean RGD = 3.4793, Non-Target mean RGD = 3.4760, Target LOO mean RGD = 3.3969.  
Cohen's d (Lure vs. Non-Target) = **+0.008**, Wilcoxon p = **0.75**.  
Lure and Non-Target are geometrically identical. Both cluster together, both separated from Target.

**Lag regression:** Mean beta = −0.011, Wilcoxon p = 0.62. Positive slopes in only 32/69 subjects.  
The geometry is a binary step function (Press vs. Withhold), not a continuous cognitive gradient.

**What this means:** The Riemannian geometry pipeline successfully detects motor preparation (press vs. withhold) but cannot separate intent violation (Lure) from novel-but-rule-compliant (Non-Target) because both share the same motor command. The 2-back task design structurally conflates the two.

## Distribution Restrictions
- `legacy/nsvd_fusion/*.npy` and `*.Xsrc.npy`: **DO NOT COMMIT** (HCP Data Use Agreement)
- `.batches/batch_ind_results/*.npy`: **DO NOT COMMIT**
- Code, results logs, datacards, and experiment documentation: freely committable

## Key Files in This Directory
| File | Description |
|---|---|
| `DATACARD.md` | This file (also see `canonical/docs/doc00nsvd_fusion_metadata.md`) |
| `data_extractionv2.py` | LaBraM weight remapping + HCP MEG loading |
| `aether_fusion.py` | Core Riemannian pipeline: EA → SPD → LOO geodesic distance |
| `baseline.py` | Face/Tool semantic baseline (59.8% LOSO proof-of-instrument) |
| `mvp0_closeout.py` | Final closeout: ordinal spacing + lag regression + motor null |
| `negative_control_matched.py` | Motor ROI negative control (matched design) |
| `v2.2_sliding_window_sweep.py` | Sliding window sweep across 250–500ms |
| `kaggle_mvp0_map.py` | Kaggle parallel MAP over 75 subjects |
| `kaggle_mvp0_reduce.py` | Kaggle REDUCE: aggregate per-subject results |
| `kaggle_batch_nsvd.py` | Batch orchestration for Kaggle runs |
| `kaggle_reanalysis.py` | Re-analysis pipeline after canonical correction |
| `c2_lag_regression.py` | OLS lag~distance regression |
| `mvp0_reanalysis.py` | Post-falsification reanalysis |
| `admissibility_check.py` | Split-half stability + min-trial admissibility filter |
| `c2_rgd_three_gate.py` | Three-gate RGD (load, friction, ordinal) |
| `c2_trial_census.py` | Trial count census across all subjects |
| `extract_lures.py` | Lure-only trial extraction utility |
| `merge_perms.py` | Merge permutation results across Kaggle batches |
| `rgd_sprint1.py` | First rapid RGD sprint implementation |
| `gate2_saliency_analysis.py` | Gate 2 saliency/feature importance |
| `c2_power_analysis.py` | Statistical power analysis |
