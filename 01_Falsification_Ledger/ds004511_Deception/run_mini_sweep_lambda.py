import os
import sys
import torch
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
# Ensure nsvd_mvp is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nsvd_mvp.sweep_phase05 import run_single_eval_05
threshold_values = [0.4, 0.5, 0.6, 0.7]
for t in threshold_values:
    print(f"\n--- Testing threshold = {t} ---")
    res = run_single_eval_05(
        lambda_l1=0.08,
        beta=0.9,
        threshold=t,
        ern_magnitude=2.0,
        noise_type='pink',
        artifact_coupling=1.0,
        distractor_correlation=0.9,
        hidden_dim=4096,
        n_samples=100
    )
    print(f"Result for threshold={t}: TPR={res['tpr']:.2f}, FPR={res['fpr']:.2f}, Separation={res['separation']:.3f}, Overall={res['overall_status']}")
    if res['overall_status'] == 'PASS':
        print(f"*** FOUND OPTIMAL THRESHOLD: {t} ***")
