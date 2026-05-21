# HE-StackingAdaBoost
Improved AdaBoost with missing-aware weighting and stacking ensemble for hydrogen embrittlement prediction in steels. MATLAB implementation.

---

## Overview

This project implements an improved AdaBoost regression model for predicting **hydrogen embrittlement (HE) susceptibility** in steels. The model features:

- **Missing‑data down‑weighting** – Penalizes imputed samples during training (λ = 0.5)
- **Stacking ensemble** – Combines multiple base learners with a linear meta‑learner
- **Dynamic loss function** – Gradually increases error sensitivity (β = 1 → 1.2)


---

## Project Structure

- `README.md` – Project documentation
- `Stacking_AdaBoostRegressor.m` – Model training function
- `predict_Stacking_AdaBoost.m` – Prediction function
- `main.m` – Example usage script


---

## How to Run the Project

### Prerequisites
- MATLAB R2020b or later
- Statistics and Machine Learning Toolbox

### Quick Start

```matlab
% Run the demo script
demo_main

% Or manually train and predict
model = Stacking_AdaBoostRegressor(X_train, y_train, mask_train, 250);
y_pred = predict_Stacking_AdaBoost(model, X_test);
```

### Input Format

| Variable | Size | Description |
|----------|------|-------------|
| `X_train` | N × D | Training features |
| `y_train` | N × 1 | Target values (RRA loss) |
| `mask_train` | N × D | Missing mask (1 = observed, 0 = imputed) |
| `X_test` | M × D | Test features |

---

## Functions Overview

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `Stacking_AdaBoostRegressor` | `X_train, y_train, mask_train, num_learners` | `model` | Trains the ensemble model |
| `predict_Stacking_AdaBoost` | `model, X_test` | `y_pred` | Generates predictions |

---

## Key Parameters

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `num_learners` | 250 | Function argument | Number of base learners |
| `lambda` | 0.5 | Line 22 in training function | Missing data penalty factor |
| `MaxNumSplits` | 4 | Line 31 in training function | Tree complexity (lower = simpler) |
| `beta` | dynamic | Line 36 in training function | Error exponent (increases with iteration) |

---

## Results (from original paper)

| Model | Test R² |
|-------|---------|
| Proposed Stacking AdaBoost | **0.70** |
| Baseline AdaBoost | 0.33 |
| Support Vector Regression | 0.31 |

> The model explains 70% of the variance in hydrogen embrittlement susceptibility.

---

## License

MIT License – free for academic and commercial use.

---
