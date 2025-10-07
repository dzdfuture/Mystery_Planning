# Main Experiment Aggregate (600 = 500 + 100)

## Model: gpt-4
### Subset 500 (mystery/generated_basic 1..500)
- count: 500
- success: 195
- success_rate: 0.3900
- time.p50: 37.4869
- time.p90: 49.8613
- time.mean: 38.0289
- attempts.mean: 1.9920
- fail_reason_top5:
  - sim.precondition_missing: 299
  - sim.goal_not_satisfied: 5
  - sim.arity_mismatch: 1

### Subset 100 (generated_basic_3 1..100)
- count: 100
- success: 82
- success_rate: 0.8200
- time.p50: 26.8409
- time.p90: 40.5474
- time.mean: 26.1507
- attempts.mean: 1.5100
- fail_reason_top5:
  - sim.precondition_missing: 15
  - sim.goal_not_satisfied: 2
  - val.fail: 1

### Combined 600
- count: 600
- success: 277
- success_rate: 0.4617
- time.p50: 36.2518
- time.p90: 49.2123
- time.mean: 36.0492
- attempts.mean: 1.9117
- fail_reason_top5:
  - sim.precondition_missing: 314
  - sim.goal_not_satisfied: 7
  - sim.arity_mismatch: 1
  - val.fail: 1

## Model: gpt-4o
### Subset 500 (mystery/generated_basic 1..500)
- count: 500
- success: 195
- success_rate: 0.3900
- time.p50: 14.1612
- time.p90: 20.3237
- time.mean: 15.2418
- attempts.mean: 2.0000
- fail_reason_top5:
  - sim.precondition_missing: 296
  - sim.goal_not_satisfied: 9

### Subset 100 (generated_basic_3 1..100)
- count: 100
- success: 82
- success_rate: 0.8200
- time.p50: 11.2529
- time.p90: 17.4190
- time.mean: 11.6432
- attempts.mean: 1.6100
- fail_reason_top5:
  - sim.precondition_missing: 12
  - sim.goal_not_satisfied: 3
  - val.fail: 3

### Combined 600
- count: 600
- success: 277
- success_rate: 0.4617
- time.p50: 13.7884
- time.p90: 19.9297
- time.mean: 14.6421
- attempts.mean: 1.9350
- fail_reason_top5:
  - sim.precondition_missing: 308
  - sim.goal_not_satisfied: 12
  - val.fail: 3

## Model: o1-mini
### Subset 500 (mystery/generated_basic 1..500)
- count: 500
- success: 201
- success_rate: 0.4020
- time.p50: 57.9464
- time.p90: 80.3264
- time.mean: 55.8557
- attempts.mean: 1.8600
- fail_reason_top5:
  - sim.goal_not_satisfied: 208
  - sim.precondition_missing: 90
  - val.fail: 1

### Subset 100 (generated_basic_3 1..100)
- count: 100
- success: 87
- success_rate: 0.8700
- time.p50: 11.6930
- time.p90: 31.9854
- time.mean: 17.4046
- attempts.mean: 1.3000
- fail_reason_top5:
  - sim.precondition_missing: 10
  - sim.goal_not_satisfied: 3

### Combined 600
- count: 600
- success: 288
- success_rate: 0.4800
- time.p50: 53.7421
- time.p90: 78.5388
- time.mean: 49.4472
- attempts.mean: 1.7667
- fail_reason_top5:
  - sim.goal_not_satisfied: 211
  - sim.precondition_missing: 100
  - val.fail: 1
