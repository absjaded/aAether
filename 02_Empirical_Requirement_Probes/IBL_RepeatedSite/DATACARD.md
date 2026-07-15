# IBL Correspondence Probe Results

Dataset: IBL RepeatedSite public OpenAlyx.
Script: `ibl_correspondence_probe.py`.
Scoring labels: `side`, `choice`, `blockPrior`, and `choice_blockPrior`.
Alignment input: unlabeled within-space neural geometry only; labels are used only after alignment for scoring.
Primary movement control: trials with movement inside `[stimOn, stimOn + 100 ms]` are dropped before spike binning.

## Corrected 100 ms label sweep

Shared command pattern:

```bash
python 02_Empirical_Requirement_Probes/IBL_RepeatedSite/ibl_correspondence_probe.py --n-mice 6 --label <label> --min-trials 300 --min-units 40 --t-post 0.1 --workers 8 --out <result_dir>
```

Locked thresholds/settings: `min_trials=300`, `min_units=40`, `dim=15`, `n_pts=400`, `t_post=0.1`.
Mechanistic change from the original brief: `t_post` was shortened from `0.4` to `0.1` seconds because the 400 ms exclusion window removed most fast-response trials. The trial threshold was restored to 300.

Each corrected run selected 6 usable sessions across 6 labs, producing 15 cross-lab pairs and 0 same-lab pairs. Selected labs: `churchlandlab_ucla`, `cortexlab`, `hausserlab`, `mainenlab`, `mrsicflogellab`, `wittenlab`.

Summary:

| label | sessions | labs | pairs | cross-lab pairs | GW edge | null edge | degeneracy | anchor 1 | anchor 2 | anchor 5 | anchor 10 | anchor 25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `side` | 6 | 6 | 15 | 15 | +0.002486 | +0.001407 | 0.949450 | 0.428662 | 0.445616 | 0.434531 | 0.432552 | 0.428769 |
| `choice` | 6 | 6 | 15 | 15 | -0.001702 | +0.013359 | 0.947703 | 0.507393 | 0.503265 | 0.482828 | 0.514620 | 0.514517 |
| `blockPrior` | 6 | 6 | 15 | 15 | +0.019528 | +0.004405 | 0.948743 | 0.474281 | 0.516576 | 0.526458 | 0.519317 | 0.539025 |
| `choice_blockPrior` | 6 | 6 | 15 | 15 | +0.011639 | +0.005674 | 0.948461 | 0.277141 | 0.302787 | 0.329855 | 0.346616 | 0.324896 |

Interpretation: the corrected IBL probe does not support a broad claim that correspondence-free alignment is impossible. It supports a narrower requirements finding: under this 100 ms pre-movement window, 6 sessions, 15-dimensional PCA, 400 GW points, and these task labels, GW did not recover a strong transferable correspondence across labs.

Label-specific readout:

- `side`: null-equivalent. Binary side is too thin to give GW useful relational structure here.
- `choice`: null-equivalent. The shuffled null was stronger than the real GW edge on average.
- `blockPrior`: strongest signal among these labels, but still weak. GW edge was only +0.019528 and degeneracy remained high. Anchors rose to 0.539025 at 25 anchors per condition, which is suggestive of some shared task/context structure but not a clean calibration curve.
- `choice_blockPrior`: richer label structure gave a small GW edge and a modest anchor rise through 10 anchors, but it remained close to null and did not become a decisive World A/B result.

Bottom line: IBL is now a requirements probe, not a possibility proof. More relational richness helps slightly, especially `blockPrior`, but this configuration still does not produce robust correspondence-free alignment.

## Historical locked run: 400 ms pre-movement window

Command:

```bash
python 02_Empirical_Requirement_Probes/IBL_RepeatedSite/ibl_correspondence_probe.py --n-mice 6 --label side --out 02_Empirical_Requirement_Probes/IBL_RepeatedSite/results
```

Original thresholds: `min_trials=300`, `min_units=40`, `dim=15`, `n_pts=400`, `t_post=0.4`.

Result: underpowered. The original 400 ms movement-exclusion window found only 2 usable sessions from 88 candidates, producing 1 cross-lab pair. Treat the printed verdict from this run as a failed/underpowered run, not as a scientific verdict.

Metrics:

| pairs | cross-lab pairs | GW edge | null edge | degeneracy | anchor 1 | anchor 2 | anchor 5 | anchor 10 | anchor 25 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | -0.001836 | +0.010034 | 0.898887 | 0.296736 | 0.317507 | 0.308605 | 0.290801 | 0.302671 |

## Historical sensitivity run: 250-trial threshold

Command:

```bash
python 02_Empirical_Requirement_Probes/IBL_RepeatedSite/ibl_correspondence_probe.py --n-mice 6 --label side --out 02_Empirical_Requirement_Probes/IBL_RepeatedSite/results_min250 --min-trials 250
```

Deviation from locked run: `min_trials` was lowered from 300 to 250 while the 400 ms window remained in place. This is retained as a sensitivity check only, because lowering the locked trial threshold was not the mechanistic fix.

Result: null-equivalent. Six usable sessions were selected across 5 labs, giving 15 session pairs, 14 cross-lab and 1 same-lab. GW edge was indistinguishable from null, and paired-anchor Procrustes did not recover the map.

Metrics:

| pairs | cross-lab pairs | same-lab pairs | GW edge | null edge | same-lab GW edge | cross-lab GW edge | degeneracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 14 | 1 | -0.005816 | -0.005395 | +0.023020 | -0.007875 | 0.924184 |

Anchor cost curve:

| anchors per condition | mean accuracy |
|---:|---:|
| 1 | 0.393989 |
| 2 | 0.404168 |
| 5 | 0.394014 |
| 10 | 0.384613 |
| 25 | 0.404634 |

## Local artifacts

- `02_Empirical_Requirement_Probes/IBL_RepeatedSite/results/ibl_correspondence_results.json`: original 400 ms locked run pair metrics.
- `02_Empirical_Requirement_Probes/IBL_RepeatedSite/results/ibl_trial_threshold_diagnostic.csv`: original 400 ms trial-count diagnostic.
- `02_Empirical_Requirement_Probes/IBL_RepeatedSite/results_min250/ibl_correspondence_results.json`: 250-trial sensitivity pair metrics.
- `02_Empirical_Requirement_Probes/IBL_RepeatedSite/results_tpost100_min300/ibl_correspondence_results.json`: corrected 100 ms `side` pair metrics.
- `02_Empirical_Requirement_Probes/IBL_RepeatedSite/results_tpost100_min300/ibl_correspondence_metadata.json`: corrected 100 ms `side` metadata.
- `02_Empirical_Requirement_Probes/IBL_RepeatedSite/results_tpost100_min300_choice/ibl_correspondence_results.json`: corrected 100 ms `choice` pair metrics.
- `02_Empirical_Requirement_Probes/IBL_RepeatedSite/results_tpost100_min300_choice/ibl_correspondence_metadata.json`: corrected 100 ms `choice` metadata.
- `02_Empirical_Requirement_Probes/IBL_RepeatedSite/results_tpost100_min300_blockPrior/ibl_correspondence_results.json`: corrected 100 ms `blockPrior` pair metrics.
- `02_Empirical_Requirement_Probes/IBL_RepeatedSite/results_tpost100_min300_blockPrior/ibl_correspondence_metadata.json`: corrected 100 ms `blockPrior` metadata.
- `02_Empirical_Requirement_Probes/IBL_RepeatedSite/results_tpost100_min300_choice_blockPrior/ibl_correspondence_results.json`: corrected 100 ms `choice_blockPrior` pair metrics.
- `02_Empirical_Requirement_Probes/IBL_RepeatedSite/results_tpost100_min300_choice_blockPrior/ibl_correspondence_metadata.json`: corrected 100 ms `choice_blockPrior` metadata.

No raw spike arrays, cache files, credentials, logs, or remote paths are part of these artifacts.