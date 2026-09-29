# Experiment Log

This file documents exploratory modelling experiments conducted during the XAI credit-risk dissertation.

These experiments were used to investigate alternative class-imbalance strategies, Random Forest sizes, and MLP configurations. They are retained for reproducibility but are not part of the final modelling pipeline unless explicitly stated.

Final reported model configurations and results are documented separately in `docs/final_experiment_summary.md`.

---

## Experiment 1 — Weighted Logistic Regression with FICO

**File:**  
`experiments/04_logistic_regression_fico_weighted.py`

**Purpose:**  
Evaluate whether class weighting could improve detection of the minority default class.

**Dataset:**  
`lc_2007_2017_ml_no_leakage_fico`

**Change from baseline:**  
Class weighting was introduced into the Logistic Regression model to give greater importance to default observations.

**Observed outcome:**  

At the standard 0.50 classification threshold:

- Default precision: 0.3979
- Default recall: 0.6541
- Default F1-score: 0.4948

At a 0.45 classification threshold:

- Default precision: 0.3716
- Default recall: 0.7462
- Default F1-score: 0.4961

**Interpretation:**  
Class weighting substantially increased default recall but reduced precision and produced more false-positive default classifications.

The experiment demonstrated the trade-off between detecting more actual defaults and incorrectly classifying additional non-default borrowers as high risk.

**Decision:**  
Not used as the final baseline Logistic Regression configuration.

The experiment was retained because it contributed to the dissertation's class-imbalance and threshold-sensitivity analysis.

---

## Experiment 2 — 500-Tree Random Forest

**File:**  
`experiments/05_500trees_random_forest.py`

**Purpose:**  
Test whether substantially increasing the number of trees improved Random Forest performance.

**Change from final configuration:**  

Final Random Forest:

`numTrees = 50`  
`maxDepth = 12`  
`seed = 42`

Experimental configuration:

`numTrees = 500`

**Observed result:**

- AUC: 0.6981
- Accuracy: 0.7453
- Default precision: 0.6382
- Default recall: 0.0354
- Default F1-score: 0.0670

The 500-tree run also encountered the model-size/resource limitation during execution and therefore was not treated as a clean final benchmark.

**Interpretation:**  
Increasing the number of trees did not provide evidence of a meaningful performance improvement and considerably increased computational requirements.

**Decision:**  
Experiment rejected.

The final Random Forest retained:

`numTrees = 50`  
`maxDepth = 12`  
`seed = 42`

---

## Experiment 3 — Deeper MLP with 128-Unit Hidden Layer

**File:**  
`experiments/06_128layer_deep_learning_mlp.py`

**Purpose:**  
Determine whether increasing MLP network capacity improved credit-default classification.

**Experimental architecture:**

`[input_size, 128, 64, 32, 2]`

**Baseline architecture:**

`[input_size, 64, 32, 2]`

**Training configuration:**  
`maxIter = 50`

**Observed outcome:**  
The additional 128-unit layer did not produce a material improvement in predictive performance and did not improve minority-class performance sufficiently to justify the additional model complexity.

**Interpretation:**  
Increasing network depth/capacity was not automatically beneficial for this dataset.

A simpler MLP architecture provided comparable performance with lower complexity.

**Decision:**  
128-unit architecture rejected.

It was excluded from the final model configuration.

---

## Experiment 4 — MLP with 50 Training Iterations

**File:**  
`experiments/06_iter50_deep_learning_mlp.py`

**Purpose:**  
Evaluate whether 50 training iterations were sufficient for the selected MLP architecture.

**Architecture:**

`[input_size, 64, 32, 2]`

**Experimental configuration:**

`maxIter = 50`

**Observed outcome:**  
The model was functional at 50 iterations, but subsequent testing showed improved overall and minority-class performance when training was extended.

**Decision:**  
The 50-iteration configuration was not retained.

The selected final MLP configuration used:

`layers = [input_size, 64, 32, 2]`  
`maxIter = 75`

---

# Final Experiment Decisions

| Experiment | Final decision | Contribution |
|---|---|---|
| Weighted Logistic Regression | Not retained as final model | Class-imbalance and threshold analysis |
| Random Forest — 500 trees | Rejected | Demonstrated that additional trees did not justify computational cost |
| MLP — 128-unit layer | Rejected | Tested effect of increased network complexity |
| MLP — 50 iterations | Rejected | Supported selection of 75 training iterations |

---

# Final Configurations Resulting from Experiments

## Random Forest

```python
numTrees = 50
maxDepth = 12
seed = 42
```

## Multilayer Perceptron

```python
layers = [input_size, 64, 32, 2]
maxIter = 75
```
