# Baseline Models (Member 2)

## Purpose
Before training any Neural Network, we measured how well simple models predict the Remaining Useful Life (RUL) of an engine. Every Neural Network in this project is reported as an improvement over these numbers.

## Setup
- Input: sliding windows of 30 cycles x 18 features, flattened to a 540-value vector because traditional models need 2-D input.
- Target: RUL in cycles. Training uses all 14,241 windows from the 80 training engines.
- Validation: one simulated pre-failure cutoff per held-out engine (20 engines), so the target is a positive remaining life as in the test set. Used for model selection only.
- Test: the last window of each of the 100 test engines, with true RUL from PM_truth.csv. Used for final reporting only, never for tuning.

## Models
1. Trivial baseline: predicts the mean training RUL (94.75 cycles) for every engine.
2. Ridge Regression: simple learned baseline. Alpha was selected on validation from {0.1, 1, 10, 100, 1000} and the best value was 1000.
3. Random Forest: 100 trees, max_features="sqrt", min_samples_leaf=3, random seed 42.

## Results (test set, 100 engines)
| Model | RMSE | MAE | R2 | NASA Score |
|---|---|---|---|---|
| Trivial (Mean RUL) | 45.79 | 36.62 | -0.214 | 73,243 |
| Ridge Regression | 26.11 | 22.42 | 0.605 | 3,200 |
| Random Forest | 25.30 | 18.94 | 0.629 | 4,534 |

## Findings
- Both learned models reduce the RMSE by about 43% (Ridge) and 45% (Random Forest) compared with the trivial baseline. The trivial model has a negative R2, so it is worse than predicting the mean of the test targets.
- Random Forest has the best RMSE and MAE and was selected on validation RMSE. Ridge has a lower NASA Score on the test set, which means Random Forest makes more of the costly late predictions. Neural Networks should therefore be judged on NASA Score as well as RMSE.

## Limitations
- The validation set has only 20 samples (one per engine), so the RMSE differences between Ridge alpha values are tiny and the alpha choice is weak.
- Random Forest results depend on the random seed. The final comparison should report mean and standard deviation over 3 seeds.
- The cycle count is one of the 18 input features.
