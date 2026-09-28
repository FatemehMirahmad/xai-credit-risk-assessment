This repository contains the technical workflow developed for the MSc Data Analytics dissertation **“Explainable Machine Learning for Credit Risk Assessment.”**

The project investigates whether more complex machine-learning models provide enough predictive improvement to justify reduced transparency in consumer credit-risk assessment.
It combines leakage controlled credit-default modelling, class-sensitive evaluation, model explainability, and complementary stakeholder evidence.

## Project Overview

The study compares four classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- Multilayer Perceptron (MLP)

Each model is evaluated under two aligned data conditions:

1. **No-FICO condition** — baseline borrower/application predictors without FICO-derived enrichment.
2. **FICO/credit-profile enriched condition** — the baseline predictors plus seven additional fields:
   - `fico_score`
   - `fico_score_missing`
   - `pub_rec_bankruptcies`
   - `pub_rec_bankruptcies_missing`
   - `has_bankruptcy`
   - `mort_acc`
   - `mort_acc_missing`

The project also applies model-specific explainability methods and Kernel SHAP, and incorporates an exploratory questionnaire on explainability, trust, classification errors, SHAP usefulness, and human oversight.

## Research Questions

1. How do Logistic Regression, Decision Tree, Random Forest, and MLP models compare in predicting credit default under no-FICO and FICO-enriched conditions?
2. What do global and local SHAP explanations reveal about the variables influencing the selected black-box model and its correct and incorrect predictions?
3. How do participants with relevant professional or analytical experience perceive explainability, trust, SHAP usefulness, classification errors, and human oversight in credit-risk assessment?
4. How can predictive performance, explanation evidence, and stakeholder perceptions be combined when assessing the suitability of machine-learning models for credit-risk decision support?

## Dataset

The modelling uses LendingClub loan data covering **2007–2017**.

Final eligible modelling sample:

- **455,318 loans**
- **364,434 training observations**
- **90,884 held-out test observations**
- Test defaults: **23,495 (25.85%)**
- Test non-defaults: **67,389 (74.15%)**

### Target definition

`default_flag = 1` for:

- Charged Off
- Default
- Late (31–120 days)
- Late (16–30 days)

`default_flag = 0` for:

- Fully Paid

Other loan statuses are excluded from the modelling sample.

### Leakage control

Variables that could leak lender decisions or post-origination outcomes are excluded from model inputs. In particular, the modelling workflow excludes variables such as:

- `grade`
- `sub_grade`
- `int_rate`
- `loan_status` as a predictor
- post-loan payment/recovery fields
- loan identifier as a model feature

The LendingClub source data is **not included in this repository**. Users should obtain the dataset separately and configure their own storage paths.

## Technology Stack

- Python
- PySpark
- Apache Spark MLlib
- Databricks
- Delta Lake
- scikit-learn / NumPy / pandas where required for downstream analysis
- SHAP
- Git / GitHub

The exact package dependencies should be installed from `requirements.txt`.

## Data Preparation

The pipeline includes:

- numeric type casting
- percentage cleaning
- categorical-value cleaning
- missing-value treatment
- employment-length extraction
- credit-history duration
- borrower-level risk flags
- engineered financial ratios
- categorical encoding with `StringIndexer` and `OneHotEncoder`
- feature assembly with `VectorAssembler`
- scaling with `StandardScaler`

Examples of engineered variables include:

- `loan_to_income_ratio`
- `installment_to_income_ratio`
- `revol_bal_to_income_ratio`
- `open_acc_to_total_acc_ratio`
- `emp_length_years`
- `credit_history_months`
- `credit_history_years`

## Databricks Tables

Key Delta tables used in the final workflow include:

```text
dissertation.lendingclub.lc_2007_2017_ml_no_leakage
dissertation.lendingclub.lc_2007_2017_ml_no_leakage_fico
dissertation.lendingclub.lc_2007_2017_final_technical_validation
```

The validation table records checks including sample sizes, split sizes, leakage exclusions, model outputs, and SHAP additivity.

## Model Configuration

The final comparison uses fixed configurations to support a controlled comparison across model families and data conditions.

| Model | Final configuration |
|---|---|
| Logistic Regression | `maxIter=50`, unweighted main model |
| Decision Tree | fitted depth = 8 |
| Random Forest | `numTrees=50`, `maxDepth=12`, `seed=42` |
| MLP | layers = `[135, 64, 32, 2]` for the enriched model |

An exploratory 500-tree Random Forest run was also tested but was not retained in the final analysis.

## Model Performance

### No-FICO condition

| Model | AUC | Accuracy | Default Recall | Default F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7106 | 0.7522 | 0.1640 | 0.2549 |
| Decision Tree | 0.6629 | 0.7453 | 0.1290 | 0.2075 |
| Random Forest | 0.7008 | 0.7463 | 0.0425 | 0.0797 |
| MLP | **0.7151** | **0.7539** | **0.1724** | **0.2659** |

### FICO/credit-profile enriched condition

| Model | AUC | Accuracy | Default Recall | Default F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.7150** | **0.7527** | **0.1692** | **0.2613** |
| Decision Tree | 0.6688 | 0.7473 | 0.1199 | 0.1970 |
| Random Forest | 0.7081 | 0.7478 | 0.0511 | 0.0948 |
| MLP | 0.7121 | 0.7524 | 0.1643 | 0.2555 |

### Main modelling result

The MLP performed slightly better than Logistic Regression in the no-FICO condition, while Logistic Regression achieved the strongest enriched AUC and default F1-score. Overall, the performance differences were small.

FICO/credit-profile enrichment produced **modest and model-dependent changes**, and default recall remained low across all final models.

## Explainability

Three levels of explainability are used.

### 1. Native / model-specific explanations

- Logistic Regression: standardized coefficients
- Decision Tree: feature importance
- Random Forest: impurity-based aggregate feature importance

### 2. Kernel SHAP for the MLP

The final Kernel SHAP configuration uses:

- training background pool: **500 observations**
- K-means background representation: **10 centres**
- global explained sample: **100 observations**
- `nsamples=200`
- enriched MLP input: **135 transformed features**

Global SHAP importance is based on mean absolute SHAP contribution across the explained sample.

Encoded components belonging to the same source variable are also grouped to provide original-variable-level interpretation.

### 3. Local diagnostic explanations

One high-confidence example is analysed for each prediction category:

- True Positive
- True Negative
- False Positive
- False Negative

These examples are illustrative and are **not intended to be representative** of all cases.

SHAP additivity checks confirmed that the four selected local explanations reconstructed the corresponding model probabilities.

### Important interpretation note

SHAP values explain the fitted model's behaviour relative to the selected background distribution. They are **not causal effects**.

Final XAI outputs are stored under:

```text
/Volumes/dissertation/lendingclub/xai_outputs
```

## Key XAI Findings

Important variables recurring across the models included:

- `fico_score`
- `dti`
- `revol_util`
- `open_acc_to_total_acc_ratio`
- `total_rev_hi_lim`
- `mort_acc`
- `home_ownership`

For the MLP, global and grouped SHAP results also highlighted variables including:

- loan purpose
- state
- recent inquiries

Cross-model comparison showed partial agreement rather than identical rankings, reinforcing that feature importance is model- and method-dependent.

## Primary Research

The dissertation also includes an anonymous questionnaire completed by **23 eligible participants** with relevant professional or analytical experience.

The questionnaire explored:

- predictive accuracy
- explainability
- trust
- SHAP usefulness
- false-negative and false-positive concerns
- consistency between explanation methods
- FICO-type information
- human oversight

Selected results:

- **87.0%** reported lower trust in an accurate model when its decisions could not be understood.
- **82.6%** considered explainability important for adverse or high-risk decisions.
- **56.5%** agreed that SHAP could make complex ML models more useful.
- **87.0%** considered false negatives a serious concern.
- **82.6%** considered false positives an important concern.
- **91.3%** agreed that human review should remain part of important credit decisions.

The questionnaire is exploratory and descriptive; the sample is not statistically representative of the wider financial sector.

## Repository Structure

```text
xai-credit-risk-assessment/
│
├── README.md
├── requirements.txt
├── config.yaml
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── explainability/
│   └── visualization/
│
├── notebooks/
├── results/
├── experiments/
├── docs/
└── dissertation/
```

The exact contents may vary as the project is maintained, but the repository separates data preparation, feature engineering, modelling, evaluation, explainability, outputs, and dissertation documentation.

## Main Workflow

The project can be reproduced conceptually in the following order:

```text
Raw LendingClub data
      then
Data cleaning
      then
Leakage control
      then
Feature engineering
      then
No-FICO / enriched modelling tables
      then
Train/test split
      then
LR / DT / RF / MLP training
      then
Model evaluation
      then
Native XAI + Kernel SHAP
      then
Cross-model explanation comparison
      then
Technical validation
      then
Integration with questionnaire findings
```

Key scripts used during development include:

```text
02_data_cleaning.py
03_feature_engineering.py
04_logistic_regression.py
05_decision_tree.py
05_random_forest.py
07_model_evaluation.py
```

Additional notebooks/scripts are used for the MLP, SHAP analysis, validation, and questionnaire analysis.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/FatemehMirahmad/xai-credit-risk-assessment.git
cd xai-credit-risk-assessment
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the data environment

The final workflow was developed in Databricks using PySpark and Delta tables.

Before running the pipeline:

- obtain the LendingClub data separately
- upload or register it in your own Databricks environment
- update paths/catalog/schema names in `config.yaml` or the relevant scripts
- ensure the required catalog/schema and storage locations exist

## Reproducibility

The final workflow uses:

- fixed train/test split seed (`42`)
- named Delta tables
- saved preprocessing/model pipelines
- fixed feature order
- stored model metrics
- explicit SHAP configuration
- saved XAI outputs
- a final technical-validation table

The final model comparison uses the same held-out test population across the no-FICO and enriched conditions.

## Limitations

Important limitations include:

- one LendingClub portfolio and historical period
- one random train/test split rather than out-of-time validation
- fixed model configurations rather than exhaustive hyperparameter optimisation
- relatively low default recall at the retained classification threshold
- Kernel SHAP approximation based on a limited background and explained sample
- model-dependent feature attribution
- exploratory questionnaire with 23 participants
- questionnaire measures stated perceptions rather than observed workplace decision behaviour

The results should therefore be interpreted as evidence from the tested specifications, not as universal conclusions about all credit-risk portfolios or model families.

## Key Conclusion

Greater model complexity did not produce a clear overall predictive advantage in this study.

Logistic Regression remained highly competitive with the MLP, while the enriched credit-profile variables produced only modest performance changes. 
The findings also show that model suitability in high-stakes credit-risk assessment should not be judged from accuracy or AUC alone.

A more complete evaluation should consider:

- default-class performance
- classification-error consequences
- explanation reliability
- transparency
- human-review requirements
- governance and documentation

## Academic Context

This repository supports an MSc Data Analytics dissertation and is intended primarily for academic, reproducibility, and portfolio purposes.

The models and outputs in this repository are **not intended for production lending decisions**.

## Author

**Fatemeh Mirahmadpour**  
MSc Data Analytics  
2026
"""
