import json
import numpy as np

ROI_NAMES_MAP = {
    1: 'L_caudalanteriorcingulate',
    2: 'L_caudalmiddlefrontal',
    6: 'L_inferiorparietal',
    25: 'L_rostralmiddlefrontal',
    26: 'L_superiorfrontal',
    29: 'L_supramarginal',
    35: 'R_caudalanteriorcingulate',
    36: 'R_caudalmiddlefrontal',
    40: 'R_inferiorparietal',
    59: 'R_rostralmiddlefrontal',
    60: 'R_superiorfrontal',
    63: 'R_supramarginal',
    24: 'L_paracentral',
    27: 'L_precentral',
    28: 'L_postcentral',
    58: 'R_paracentral',
    61: 'R_precentral',
    62: 'R_postcentral'
}

def get_roi_name(idx):
    return ROI_NAMES_MAP.get(idx, f"ROI_{idx}")

def unvech(vector, n):
    matrix = np.zeros((n, n))
    idx = 0
    for i in range(n):
        matrix[i, i] = vector[idx]
        idx += 1
        for j in range(i + 1, n):
            matrix[i, j] = vector[idx] / np.sqrt(2)
            matrix[j, i] = matrix[i, j]
            idx += 1
    return matrix

results_file = "results/proxy_baseline_results.jsonl"
coefs = []
with open(results_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        data = json.loads(line)
        coef = np.array(data["best_coef"])
        coefs.append(coef)

avg_coef = np.mean(coefs, axis=0)
tangent_matrix = unvech(avg_coef, 68)
roi_saliency = np.abs(tangent_matrix).sum(axis=1)

sorted_indices = np.argsort(roi_saliency)[::-1]

print("=== Gate 2: ROI Saliency Rankings ===")
for rank, idx in enumerate(sorted_indices[:15], 1):
    print(f"Rank {rank:2d}: {get_roi_name(idx):>30} (Index {idx:2d}) - Saliency: {roi_saliency[idx]:.4f}")

l_caudal_rank = np.where(sorted_indices == 2)[0][0] + 1
print(f"\nL_caudalmiddlefrontal (Left DLPFC) Rank: {l_caudal_rank}")
if l_caudal_rank <= 3:
    print("[PASS] Left DLPFC prominence confirmed.")
else:
    print("[FAIL] Left DLPFC not in top-3.")
