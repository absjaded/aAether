#!/usr/bin/env python3
"""
IBL cross-session correspondence probe.

Question: can two neural representational spaces with no shared units and no
paired trials be aligned from relational geometry alone?

Design:
- Use IBL RepeatedSite sessions across labs.
- Build a separate latent space for each session.
- Fit correspondence-free GW-style alignments without trial labels.
- Use labels only after alignment for scoring.
- Report label transfer, anchor sensitivity, and coupling degeneracy.

Interpretation:
- Strong correspondence-free recovery would support geometry-only alignment.
- Recovery only after anchors estimates calibration cost.
- Null-equivalent recovery means this public neural geometry is not sufficient
  for the intended alignment claim under the tested settings.

Usage:
    pip install ONE-api POT scikit-learn numpy
    python ibl_correspondence_probe.py --n-mice 6 --label side --t-post 0.1 --workers 8 --out artifacts/ibl_tpost100_min300/
"""
import argparse, json, os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')

SEED = 42
rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------
# 1. LOAD: bin spikes into trial x neuron matrices. Raw spikes never persisted.
# --------------------------------------------------------------------------
def load_session(one, eid, t_pre=0.0, t_post=0.4, min_units=40, min_trials=300):
    """
    Returns X (trials x units) spike counts in a window locked to stimulus onset,
    plus the trial label table. Window is PRE-MOVEMENT by construction:
    [stimOn, stimOn + t_post] and trials with earlier movement are dropped, so the
    population vector is not dominated by the motor response.
    (LAW 1 discipline, imported. Yes, it matters even in mice.)
    """
    tr = one.load_object(eid, 'trials')
    n = len(tr['choice'])
    stim_on = np.asarray(tr['stimOn_times'], float)
    first_mv = np.asarray(tr.get('firstMovement_times', np.full(n, np.nan)), float)

    block_prior = np.asarray(tr['probabilityLeft'], float)
    labels = dict(
        choice        = np.asarray(tr['choice'], float),
        contrastLeft  = np.nan_to_num(np.asarray(tr['contrastLeft'], float)),
        contrastRight = np.nan_to_num(np.asarray(tr['contrastRight'], float)),
        feedbackType  = np.asarray(tr['feedbackType'], float),
        blockPrior    = np.asarray([f"{b:.3f}" for b in block_prior]),
    )
    # stimulus side: the cleanest ground truth for scoring the coupling
    labels['side'] = np.where(labels['contrastLeft'] > 0, -1.0,
                      np.where(labels['contrastRight'] > 0, 1.0, 0.0))
    labels['choice_blockPrior'] = np.asarray([
        f"{int(c)}:{b:.3f}" for c, b in zip(labels['choice'], block_prior)
    ])

    # keep trials with a valid stimOn, a real choice, and NO movement inside the window
    ok = (~np.isnan(stim_on)) & (labels['choice'] != 0)
    ok &= np.isnan(first_mv) | (first_mv > stim_on + t_post)
    if ok.sum() < min_trials:
        return None

    # spikes
    coll = None
    for c in ['alf/probe00/pykilosort', 'alf/probe01/pykilosort',
              'alf/probe00', 'alf/probe01']:
        try:
            sp = one.load_object(eid, 'spikes', collection=c,
                                 attribute=['times', 'clusters'])
            cl = one.load_object(eid, 'clusters', collection=c,
                                 attribute=['metrics'])
            coll = c; break
        except Exception:
            continue
    if coll is None:
        return None

    st, sc = np.asarray(sp['times']), np.asarray(sp['clusters'])

    # quality: keep only good units (IBL ships a QC label)
    try:
        m = cl['metrics']
        good = np.asarray(m['label'] >= 1.0) if 'label' in m else None
        keep_u = np.where(good)[0] if good is not None else np.unique(sc)
    except Exception:
        keep_u = np.unique(sc)
    if len(keep_u) < min_units:
        return None

    uid = {u: i for i, u in enumerate(keep_u)}
    idx = np.where(ok)[0]
    X = np.zeros((len(idx), len(keep_u)), dtype=np.float32)

    order = np.argsort(st)
    st, sc = st[order], sc[order]
    for r, t in enumerate(idx):
        lo, hi = stim_on[t] + t_pre, stim_on[t] + t_post
        a, b = np.searchsorted(st, lo), np.searchsorted(st, hi)
        u, c = np.unique(sc[a:b], return_counts=True)
        for uu, cc in zip(u, c):
            j = uid.get(uu)
            if j is not None:
                X[r, j] = cc

    lab = {k: v[idx] for k, v in labels.items()}
    return dict(X=X, labels=lab, n_units=len(keep_u), n_trials=len(idx), coll=coll)


def trial_count_session(one, eid, t_post=0.4):
    """Count valid pre-movement trials without loading spikes."""
    tr = one.load_object(eid, 'trials')
    n = len(tr['choice'])
    stim_on = np.asarray(tr['stimOn_times'], float)
    first_mv = np.asarray(tr.get('firstMovement_times', np.full(n, np.nan)), float)
    choice = np.asarray(tr['choice'], float)
    ok = (~np.isnan(stim_on)) & (choice != 0)
    ok &= np.isnan(first_mv) | (first_mv > stim_on + t_post)
    return int(ok.sum())


# --------------------------------------------------------------------------
# 2. LATENT: each brain gets its own space. No shared basis. That is the point.
# --------------------------------------------------------------------------
def to_latent(X, dim=15):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(np.sqrt(X))          # variance-stabilise counts
    d = min(dim, Z.shape[1] - 1, Z.shape[0] - 1)
    return PCA(n_components=d, random_state=SEED).fit_transform(Z)


# --------------------------------------------------------------------------
# 3. THE ALIGNMENTS
# --------------------------------------------------------------------------
def gw_align(Za, Zb, n=400, eps=5e-3, rng_obj=None):
    """
    Gromov-Wasserstein: align two spaces using ONLY intra-space distance geometry.
    NO paired points. NO labels. This is the World B test.
    Returns the coupling and its degeneracy (entropy ratio; 1.0 = fully uniform
    = the plan learned nothing, even if downstream accuracy looks okay).
    """
    import ot
    ia = rng.choice(len(Za), min(n, len(Za)), replace=False)
    ib = rng.choice(len(Zb), min(n, len(Zb)), replace=False)
    A, B = Za[ia], Zb[ib]
    Ca = ot.dist(A, A); Cb = ot.dist(B, B)
    Ca /= Ca.max(); Cb /= Cb.max()
    p = ot.unif(len(A)); q = ot.unif(len(B))
    T = ot.gromov.entropic_gromov_wasserstein(Ca, Cb, p, q, 'square_loss',
                                              epsilon=eps, max_iter=1000)
    Tn = T / (T.sum() + 1e-30)
    H = -(Tn * np.log(Tn + 1e-30)).sum()
    Hmax = np.log(Tn.size)
    return T, ia, ib, float(H / Hmax)       # degeneracy in [0,1]; ->1 is degenerate


def procrustes_anchors(Za, Zb, ya, yb, k, rng_obj=None):
    """
    World A: how few PAIRED anchors does an orthogonal map need?
    Anchors are trials matched on the ground-truth condition. k per condition.
    """
    from scipy.linalg import orthogonal_procrustes
    conds = np.unique(ya)
    A, B = [], []
    for c in conds:
        ai = np.where(ya == c)[0]; bi = np.where(yb == c)[0]
        m = min(k, len(ai), len(bi))
        if m == 0: continue
        A.append(Za[rng.choice(ai, m, replace=False)].mean(0))
        B.append(Zb[rng.choice(bi, m, replace=False)].mean(0))
    if len(A) < 2: return None
    A, B = np.array(A), np.array(B)
    d = min(Za.shape[1], Zb.shape[1])
    R, _ = orthogonal_procrustes(Za[:, :d].mean(0, keepdims=True) * 0 + A[:, :d],
                                 B[:, :d])
    return R, d


# --------------------------------------------------------------------------
# 4. SCORING: labels enter ONLY here, after the fact.
# --------------------------------------------------------------------------
def score_coupling(T, ya, yb):
    """Does the recovered coupling map like-condition to like-condition?"""
    m = T.argmax(1)
    return float((ya == yb[m]).mean())

def chance(ya, yb):
    va, ca = np.unique(ya, return_counts=True)
    vb, cb = np.unique(yb, return_counts=True)
    pa = {v.item() if hasattr(v, 'item') else v: c / len(ya) for v, c in zip(va, ca)}
    pb = {v.item() if hasattr(v, 'item') else v: c / len(yb) for v, c in zip(vb, cb)}
    return float(sum(pa.get(k, 0.0) * pb.get(k, 0.0) for k in set(pa) | set(pb)))


def score_pair(args):
    i, j, A, B, n_pts, seed = args
    pair_rng = np.random.default_rng(seed)
    same = A['lab'] == B['lab']
    T, ia, ib, degen = gw_align(A['Z'], B['Z'], n=n_pts, rng_obj=pair_rng)
    ya, yb = A['y'][ia], B['y'][ib]
    acc = score_coupling(T, ya, yb)
    ch = chance(ya, yb)

    Zb_sh = B['Z'][pair_rng.permutation(len(B['Z']))]
    Tn, ian, ibn, _ = gw_align(A['Z'], Zb_sh, n=n_pts, rng_obj=pair_rng)
    acc_null = score_coupling(Tn, A['y'][ian], B['y'][ibn])

    anch = {}
    for k in [1, 2, 5, 10, 25]:
        r = procrustes_anchors(A['Z'], B['Z'], A['y'], B['y'], k, rng_obj=pair_rng)
        if r is None:
            continue
        R, d = r
        Zm = A['Z'][:, :d] @ R
        from sklearn.neighbors import KNeighborsClassifier
        kn = KNeighborsClassifier(5).fit(B['Z'][:, :d], B['y'])
        anch[k] = float((kn.predict(Zm) == A['y']).mean())

    row = dict(a=A['lab'], b=B['lab'], same_lab=bool(same),
               gw_acc=acc, gw_null=acc_null, chance=ch,
               gw_edge=acc-ch, degeneracy=degen, anchors=anch)
    tag = "SAME-LAB " if same else "CROSS-LAB"
    line = (f"{tag} {A['lab'][:12]:12} <-> {B['lab'][:12]:12} | "
            f"GW {acc:.3f} (chance {ch:.3f}, null {acc_null:.3f}) | "
            f"degen {degen:.3f} | anchors " +
            " ".join(f"{k}:{v:.2f}" for k, v in anch.items()))
    return i, j, row, line


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-mice', type=int, default=6)
    ap.add_argument('--dim', type=int, default=15)
    ap.add_argument('--n-pts', type=int, default=400)
    ap.add_argument('--label', default='side', choices=['side','choice','blockPrior','choice_blockPrior'])
    ap.add_argument('--min-trials', type=int, default=300)
    ap.add_argument('--min-units', type=int, default=40)
    ap.add_argument('--t-post', type=float, default=0.4,
                    help='Seconds after stimOn for the pre-movement spike window.')
    ap.add_argument('--workers', type=int, default=1,
                    help='Parallel workers for pair scoring; keep below physical cores.')
    ap.add_argument('--diagnostic-only', action='store_true',
                    help='Count pre-movement trials per session without loading spikes.')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from one.api import ONE
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True)

    eids = one.search(datasets=['spikes.times.npy'], tag='2024_Q2_IBL_et_al_RepeatedSite')
    print(f"pre-movement window: [stimOn, stimOn + {a.t_post:.3f}s]")
    # Spread across labs, but select AFTER the session passes the fixed thresholds.
    # This fixes the zero-usable-session failure without changing analysis knobs.
    by_lab = {}
    for e in eids:
        lab = one.get_details(e)['lab']
        by_lab.setdefault(lab, []).append(e)
    labs = list(by_lab)
    print(f"candidate sessions: {len(eids)} across {len(labs)} labs")

    if a.diagnostic_only:
        diag_rows = []
        for lab in labs:
            for eid in by_lab[lab]:
                print(f"diagnostic {lab} {str(eid)[:8]} ...", flush=True)
                try:
                    n_trials = trial_count_session(one, eid, t_post=a.t_post)
                except Exception as exc:
                    diag_rows.append(dict(lab=lab, eid=str(eid), n_trials=None, error=str(exc)))
                    print(f"   error: {exc}")
                    continue
                diag_rows.append(dict(lab=lab, eid=str(eid), n_trials=n_trials, error=''))
                print(f"   {n_trials} pre-movement trials")
        import csv
        diag_path = os.path.join(a.out, 'ibl_trial_threshold_diagnostic.csv')
        with open(diag_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['lab', 'eid', 'n_trials', 'error'])
            w.writeheader(); w.writerows(diag_rows)
        passing = [r for r in diag_rows if isinstance(r['n_trials'], int) and r['n_trials'] >= a.min_trials]
        passing_labs = sorted(set(r['lab'] for r in passing))
        print(f"diagnostic sessions >= {a.min_trials} trials: {len(passing)} across {len(passing_labs)} labs")
        print(f"saved -> {diag_path}")
        return

    S = []
    used_by_lab = {lab: 0 for lab in labs}
    skip_count = 0
    round_i = 0
    while len(S) < a.n_mice:
        progressed = False
        for lab in labs:
            if used_by_lab[lab] >= 2:
                continue
            es = by_lab[lab]
            if round_i >= len(es):
                continue
            progressed = True
            eid = es[round_i]
            print(f"loading {lab} {str(eid)[:8]} ...", flush=True)
            d = load_session(one, eid, t_post=a.t_post, min_units=a.min_units, min_trials=a.min_trials)
            if d is None:
                skip_count += 1
                print("   skipped (insufficient units/trials)")
                continue
            d['lab'] = lab; d['eid'] = str(eid)
            d['Z'] = to_latent(d['X'], a.dim)
            d['y'] = d['labels'][a.label]
            d.pop('X', None); d.pop('labels', None)
            print(f"   accepted: {d['n_trials']} pre-movement trials x {d['n_units']} good units")
            S.append(d)
            used_by_lab[lab] += 1
            if len(S) >= a.n_mice:
                break
        if not progressed:
            break
        round_i += 1
    print(f"selected {len(S)} usable sessions across {len(set(s['lab'] for s in S))} labs; skipped {skip_count} candidates\n")

    print(f"\nusable sessions: {len(S)}")
    print(f"ground-truth label for scoring: '{a.label}' (NEVER shown to the aligner)\n")

    rows = []
    pair_args = [(i, j, S[i], S[j], a.n_pts, SEED + i * 1009 + j)
                 for i in range(len(S)) for j in range(i + 1, len(S))]
    if a.workers > 1 and len(pair_args) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        worker_count = min(a.workers, len(pair_args))
        print(f"pair scoring workers: {worker_count}")
        completed = []
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            futs = [ex.submit(score_pair, p) for p in pair_args]
            for fut in as_completed(futs):
                completed.append(fut.result())
        for _, _, row, line in sorted(completed, key=lambda x: (x[0], x[1])):
            rows.append(row)
            print(line)
    else:
        print("pair scoring workers: 1")
        for p in pair_args:
            _, _, row, line = score_pair(p)
            rows.append(row)
            print(line)

    # ---------------- VERDICT ----------------
    gw   = np.array([r['gw_edge'] for r in rows])
    nl   = np.array([r['gw_null'] - r['chance'] for r in rows])
    dg   = np.array([r['degeneracy'] for r in rows])
    xl   = np.array([not r['same_lab'] for r in rows])

    print("\n" + "="*66)
    print("VERDICT")
    print("="*66)
    print(f"  pairs: {len(rows)}  ({xl.sum()} cross-lab)")
    print(f"  GW edge over chance : {gw.mean():+.3f}   (null: {nl.mean():+.3f})")
    if xl.any() and (~xl).any():
        print(f"     same-lab {gw[~xl].mean():+.3f}  |  cross-lab {gw[xl].mean():+.3f}")
    print(f"  coupling degeneracy : {dg.mean():.3f}  (1.0 = uniform = learned nothing)")

    best_anchor = {}
    for r in rows:
        for k,v in r['anchors'].items():
            best_anchor.setdefault(k, []).append(v)
    if best_anchor:
        print("\n  paired-anchor Procrustes (World A cost curve):")
        for k in sorted(best_anchor):
            print(f"     {k:2d} anchors/condition -> {np.mean(best_anchor[k]):.3f}")

    gw_works = gw.mean() > 0.05 and gw.mean() > nl.mean() + 0.03 and dg.mean() < 0.97
    anchors_work = bool(best_anchor) and np.mean(best_anchor[max(best_anchor)]) > 0.55

    print()
    if gw_works:
        print("  >> WORLD B plausible: correspondence-free alignment recovers real")
        print("     structure across brains with no paired points. Confirm on real")
        print("     human/model data before claiming a business model.")
    elif anchors_work:
        print("  >> WORLD A: GW is weak/degenerate, but a few paired anchors recover")
        print("     the map. Calibrated-instrument company. The anchor count above IS")
        print("     the per-domain calibration cost.")
    else:
        print("  >> NULL UNDER THIS SPEC: real GW matches the null and anchors are weak")
        print("     for this window, label, trial count, dimensionality, and SNR.")
        print("     Treat this as a requirements result, not as proof that")
        print("     correspondence-free alignment is impossible in general.")
    print("="*66)

    with open(os.path.join(a.out, 'ibl_correspondence_results.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=float)
    meta = dict(seed=SEED, n_mice=a.n_mice, dim=a.dim, n_pts=a.n_pts,
                label=a.label, min_trials=a.min_trials, min_units=a.min_units,
                t_post=a.t_post, workers=a.workers, selected_sessions=len(S), skipped_candidates=skip_count,
                labs=sorted(set(s['lab'] for s in S)))
    with open(os.path.join(a.out, 'ibl_correspondence_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\nsaved -> {a.out}/ibl_correspondence_results.json")
    print(f"saved -> {a.out}/ibl_correspondence_metadata.json")


if __name__ == '__main__':
    main()

