# Extended Analysis (All data)

## Model: gpt-4
- n: 600, success: 277, success_rate: 0.4617
- success_rate_ci95: [0.4221, 0.5017]
- time: median=36.25s, p90=49.21s, trimmed_mean_5pct=36.09s
- plan_structure_failed: empty_len0=0, near_empty_len1=0, empty_rate=0.0000, near_empty_rate=0.0000
- retry_benefit_curve (preview): 1:0.088, 2:0.462
- failure_step_counts: [[1, 255], [2, 60], [3, 160], [4, 9], [5, 28], [6, 7]]
- hazard_per_step: [[1, 0.4913294797687861], [2, 0.22727272727272727], [3, 0.7843137254901961], [4, 0.20454545454545456], [5, 0.8], [6, 1.0]]
- top_npmi_action_pred_pairs:
  - (unstack ↔ handempty): npmi=0.927, count=2
  - (feast ↔ craves): npmi=0.807, count=48
  - (stack ↔ clear): npmi=0.782, count=11
  - (pick-up ↔ ontable): npmi=0.763, count=5
  - (stack ↔ holding): npmi=0.701, count=3
  - (pick-up ↔ clear): npmi=0.677, count=9
  - (succumb ↔ pain): npmi=0.661, count=16
  - (overcome ↔ pain): npmi=0.461, count=36
  - (pick-up ↔ handempty): npmi=0.391, count=1
  - (attack ↔ planet): npmi=0.282, count=67
- signal_presence(failed): sim=322, translate=0, val=323
- signal_overlaps(failed): sim&trans=0, sim&val=322, trans&val=0, all3=0

## Model: gpt-4o
- n: 600, success: 277, success_rate: 0.4617
- success_rate_ci95: [0.4221, 0.5017]
- time: median=13.79s, p90=19.93s, trimmed_mean_5pct=14.28s
- plan_structure_failed: empty_len0=0, near_empty_len1=0, empty_rate=0.0000, near_empty_rate=0.0000
- retry_benefit_curve (preview): 1:0.065, 2:0.462
- failure_step_counts: [[1, 76], [2, 54], [3, 290], [4, 29], [5, 68], [6, 1], [7, 1]]
- hazard_per_step: [[1, 0.1464354527938343], [2, 0.12189616252821671], [3, 0.7455012853470437], [4, 0.29292929292929293], [5, 0.9714285714285714], [6, 0.5], [7, 1.0]]
- top_npmi_action_pred_pairs:
  - (attack ↔ planet): npmi=0.917, count=62
  - (pick-up ↔ clear): npmi=0.795, count=16
  - (stack ↔ holding): npmi=0.749, count=5
  - (stack ↔ clear): npmi=0.651, count=11
  - (feast ↔ craves): npmi=0.628, count=15
  - (pick-up ↔ handempty): npmi=0.605, count=2
  - (overcome ↔ pain): npmi=0.600, count=273
  - (unstack ↔ clear): npmi=0.519, count=2
  - (feast ↔ harmony): npmi=0.458, count=4
  - (feast ↔ province): npmi=0.370, count=37
- signal_presence(failed): sim=320, translate=0, val=323
- signal_overlaps(failed): sim&trans=0, sim&val=320, trans&val=0, all3=0

## Model: o1-mini
- n: 600, success: 288, success_rate: 0.4800
- success_rate_ci95: [0.4403, 0.5200]
- time: median=53.74s, p90=78.54s, trimmed_mean_5pct=49.04s
- plan_structure_failed: empty_len0=165, near_empty_len1=0, empty_rate=0.5288, near_empty_rate=0.0000
- retry_benefit_curve (preview): 1:0.233, 2:0.480
- failure_step_counts: [[1, 20], [2, 11], [3, 48], [4, 13], [5, 36], [6, 9]]
- hazard_per_step: [[1, 0.145985401459854], [2, 0.09401709401709402], [3, 0.4528301886792453], [4, 0.22413793103448276], [5, 0.8], [6, 1.0]]
- top_npmi_action_pred_pairs:
  - (pick-up ↔ handempty): npmi=0.740, count=2
  - (put-down ↔ holding): npmi=0.718, count=1
  - (stack ↔ holding): npmi=0.703, count=3
  - (pick-up ↔ clear): npmi=0.656, count=4
  - (feast ↔ craves): npmi=0.630, count=5
  - (stack ↔ clear): npmi=0.612, count=4
  - (succumb ↔ pain): npmi=0.572, count=7
  - (unstack ↔ clear): npmi=0.553, count=1
  - (attack ↔ planet): npmi=0.506, count=12
  - (overcome ↔ pain): npmi=0.262, count=18
- signal_presence(failed): sim=311, translate=0, val=312
- signal_overlaps(failed): sim&trans=0, sim&val=311, trans&val=0, all3=0

## Pairwise success rate tests (two-proportion z-test)
- gpt-4_vs_gpt-4o: p1=0.4617, p2=0.4617, diff=0.0000, z=0.000, p=1.0000
- gpt-4_vs_o1-mini: p1=0.4617, p2=0.4800, diff=-0.0183, z=-0.636, p=0.5247
- gpt-4o_vs_o1-mini: p1=0.4617, p2=0.4800, diff=-0.0183, z=-0.636, p=0.5247
