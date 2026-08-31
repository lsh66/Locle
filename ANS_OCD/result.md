(Locle) PS C:\Users\23739\Desktop\study\work2\Locle> python -m ANS_OCD.main --dataset cora
[ANS-OCD] Running on Cora dataset (fixed SBERT split).
[Stage 2] Subspace clustering and K estimation ...
  K_init estimated by Eigengap: 1
  LLM validated K: 2
  Themes: {1: 'Neural Networks and Learning Algorithms', 2: 'Machine Learning and Statistical Models'}
  Merge suggestions: []

[Stage 6] Self-training round 1/3 ...
  High-entropy nodes selected: 271
  Expanded nodes per community: {0: 18}
  Conflicts detected: 0

[Stage 6] Self-training round 2/3 ...
  High-entropy nodes selected: 271
  Expanded nodes per community: {0: 18}
  Conflicts detected: 0
  Mean probability change: 0.029074

[Stage 6] Self-training round 3/3 ...
  High-entropy nodes selected: 271
  Expanded nodes per community: {0: 18}
  Conflicts detected: 0
  Mean probability change: 0.005243

[ANS-OCD] Final overlapping communities:
  Community 0: 31 nodes
  Community 1: 0 nodes

[ANS-OCD] Evaluation metrics:
  omega_index: 0.9043
  avg_jaccard: 0.6222
  f1_micro: 0.8100
  f1_macro: 0.4475
  avg_precision: 0.0000
  avg_recall: 0.0000
  avg_f1: 0.0000
  nmi_flat: 0.0046
  ari_flat: -0.0086
  avg_pred_communities: 0.0114
  avg_true_communities: 0.3685
  overlap_rate: 0.0000

ANS-OCD completed. Final membership shape: (2708, 2)