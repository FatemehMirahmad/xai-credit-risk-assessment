## Data Sources
lc_loan.csv and lc_2016_2017.csv from https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans?select=lc_loan.csv
accepted_2007_to_2018Q4 from https://www.kaggle.com/datasets/wordsforthewise/lending-club/data
## Train/Test Split

Method:

`randomSplit([0.8, 0.2], seed=42)`
The four final models are:

- Logistic Regression
- Decision Tree
- Random Forest
- Multilayer Perceptron (MLP)

Two modelling conditions are compared:

1. **No-FICO condition**
2. **FICO-enriched condition** with seven extra features:
                                                          `fico_score`
                                                          `fico_score_missing`
                                                          `pub_rec_bankruptcies`
                                                          `pub_rec_bankruptcies_missing`
                                                          `has_bankruptcy`
                                                          `mort_acc`
                                                          `mort_acc_missing`


## Target

Binary target variable: `default_flag`
- **Non-default** (Fully Paid) = 0
- **Default** (Default, Charged Off, Late (31-120 days), Late (16-30 days)) = 1

## Leakage-Control Strategy

The following variables were excluded from final modelling:
`grade`
`sub_grade`
`int_rate`
These variables represent LendingClub's own assessment and pricing of borrower risk and were therefore treated as inappropriate inputs for the final leakage-controlled comparison.

Post-loan outcome variables (Like: `total_pymnt` `recoveries` `last_pymnt_amnt`) were also excluded from predictive modelling. 

Other raw fields used only for feature derivation were also excluded from the final predictor set. Examples include:
`loan_status`
`term`
`emp_length`
`earliest_cr_line`
`issue_d`
`fico_range_low`
`fico_range_high`
`id`

## Main Engineered Features

The feature-engineering stage created additional borrower and credit-risk indicators.
`loan_to_income_ratio`
`installment_to_income_ratio`
`revol_bal_to_income_ratio`
`open_acc_to_total_acc_ratio`
`emp_length_years`
`credit_history_months`
`credit_history_years`
`has_delinquency_2yrs`
`has_public_record`
`has_collections_12_mths`
`has_mths_since_last_delinq`
`has_bankruptcy`

## Missing-Value

Missing values were handled according to the type and interpretation of each variable. Examples include:
`emp_length_years_missing`
`mths_since_rcnt_il_missing`
`il_util_missing`
`tot_coll_amt_missing`
`tot_cur_bal_missing`
`total_rev_hi_lim_missing`
`fico_score_missing`
`pub_rec_bankruptcies_missing`
`mort_acc_missing`
For selected variables, a value of -1 was used to represent unavailable information after a separate missingness indicator had been created where appropriate.

Other continuous variables were filled using approximate medians.
Categorical missing values were handled consistently where required.
Categorical inconsistencies were also standardised during data cleaning. For example, applications types such as `Individual` `INDIVIDUAL`  were standardised to `INDIVIDUAL` and
`Joint App` and `JOINT` were standardised to`JOINT`.

Final counts:

- Final eligible observations: 455,318
- Training observations: 364,434
- Test observations: 90,884

Training target distribution:

- Non-default: 271,046
- Default: 93,388

Test target distribution:

- Non-default: 67,389
- Default: 23,495

The complete dataset contained 338,411 non-default observations
(74.15%) and 116,866 default observations (25.85%).


## Final Models
Four models were included in the final comparison.

### Logistic Regression

- `maxIter = 50`
Logistic Regression was included as the main linear benchmark.
It provides relatively strong intrinsic interpretability because model coefficients indicate both:
1. magnitude of association; and
2. direction of association with the predicted default outcome.
Categorical variables were indexed and one-hot encoded.
Numerical inputs were assembled and scaled before final Logistic Regression training.


### Decision Tree

- `maxDepth = 8`
A Decision Tree classifier was included as an intrinsically interpretable nonlinear model.
Its interpretation is available through:
tree structure;
decision rules; and
feature importance.
Unlike Logistic Regression coefficients, tree feature-importance values do not indicate whether a feature increases or decreases predicted default risk.

### Random Forest

- `numTrees = 50`
- `maxDepth = 12`
Random Forest was included as an ensemble tree-based model.
It provides more flexible nonlinear modelling than a single Decision Tree while retaining a global feature-importance measure.
The final experiment retained the main Random Forest specification used in the project.
An additional experiment using a substantially larger number of trees was also conducted during model development, but it did not produce a sufficiently meaningful performance improvement to justify replacing the simpler final configuration.

### Multilayer Perceptron

- Architecture: `[135, 64, 32, 2]`
- `maxIter = 50`
- `blockSize = 256`
- `solver = l-bfgs`
The Multilayer Perceptron was included as the main neural-network / deep-learning model.
The transformed vector contains:
numerical predictors;
missingness indicators; and
one-hot encoded categorical predictors.
Because the MLP does not provide a simple intrinsic feature-importance interpretation, its final explanations were produced using Kernel SHAP.

## Model Evaluation Metrics

The final comparison uses several complementary metrics.
Overall / threshold-independent performance
AUC
Accuracy
Weighted Precision
Weighted Recall
Weighted F1
Default-class performance
Default Precision
Default Recall
Default F1

The confusion matrix was also examined.
These metrics were retained because accuracy alone can give an incomplete picture when the target classes are imbalanced.
In particular:

- Default precision measures how many loans predicted as defaults actually defaulted.
- Default recall measures how many actual defaults were identified by the model.
- Default F1 balances default precision and recall.
- AUC evaluates the ability of the model to discriminate between the two classes across classification thresholds.

## Final No-FICO Results

| Model                 |        AUC |   Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Default Precision | Default Recall | Default F1 |
| --------------------- | ---------: | ---------: | -----------------: | --------------: | ----------: | ----------------: | -------------: | ---------: |
| Logistic Regression   |     0.7106 |     0.7522 |             0.7163 |          0.7522 |      0.6972 |            0.5721 |         0.1640 |     0.2549 |
| Decision Tree         |     0.6629 |     0.7453 |             0.7005 |          0.7453 |      0.6826 |            0.5305 |         0.1290 |     0.2075 |
| Random Forest         |     0.7008 |     0.7463 |             0.7202 |          0.7463 |      0.6530 |            0.6402 |         0.0425 |     0.0797 |
| Multilayer Perceptron | **0.7151** | **0.7539** |             0.7197 |      **0.7539** |  **0.7006** |            0.5805 |     **0.1724** | **0.2659** |

## Final FICO-Enriched Results

| Model                 |        AUC |   Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Default Precision | Default Recall | Default F1 |
| --------------------- | ---------: | ---------: | -----------------: | --------------: | ----------: | ----------------: | -------------: | ---------: |
| Logistic Regression   | **0.7150** | **0.7527** |             0.7173 |      **0.7527** |  **0.6989** |            0.5734 |     **0.1692** | **0.2613** |
| Decision Tree         |     0.6688 |     0.7473 |             0.7055 |          0.7473 |      0.6813 |            0.5522 |         0.1199 |     0.1970 |
| Random Forest         |     0.7081 |     0.7478 |         **0.7254** |          0.7478 |      0.6573 |        **0.6557** |         0.0511 |     0.0948 |
| Multilayer Perceptron |     0.7121 |     0.7524 |             0.7169 |          0.7524 |      0.6974 |            0.5741 |         0.1643 |     0.2555 |

Logistic Regression provides substantially greater intrinsic interpretability while achieving comparable predictive performance.

## Effect of FICO Enrichment

| Model                 | Change in AUC | Change in Accuracy | Change in Weighted F1 | Change in Default Recall | Change in Default F1 |
| --------------------- | ------------: | -----------------: | --------------------: | -----------------------: | -------------------: |
| Logistic Regression   |       +0.0044 |            +0.0005 |               +0.0017 |                  +0.0052 |              +0.0064 |
| Decision Tree         |       +0.0059 |            +0.0020 |               -0.0013 |                  -0.0091 |              -0.0105 |
| Random Forest         |       +0.0073 |            +0.0015 |               +0.0043 |                  +0.0086 |              +0.0151 |
| Multilayer Perceptron |       -0.0030 |            -0.0015 |               -0.0032 |                  -0.0081 |              -0.0104 |

FICO enrichment therefore did not improve every model or every metric.
Logistic Regression improved modestly.
Random Forest also improved in AUC and default-class metrics.
Decision Tree improved in AUC but experienced weaker default-class performance.
The MLP performed slightly worse after enrichment.
The results therefore do not support the assumption that adding FICO-related variables automatically produces a substantially stronger model.
Across both feature conditions, Logistic Regression and the Multilayer Perceptron formed a narrow leading group.
- Without FICO enrichment:
MLP AUC = 0.7151
LR AUC  = 0.7106

- With FICO enrichment:
LR AUC  = 0.7150
MLP AUC = 0.7121
Logistic Regression provides substantially greater intrinsic interpretability while achieving comparable predictive performance.

## Class imbalance and decision-threshold
Class imbalance and decision-threshold behaviour were investigated separately as a supplementary experiment.
The main final comparison was not replaced by a class-weighted model.
The natural held-out sample remained the basis of the primary comparison.
A separate weighted Logistic Regression / threshold analysis was retained to demonstrate that changes in class weighting or classification threshold can substantially alter the trade-off between:

- Default Precision
- Default Recall

This supplementary experiment is useful for demonstrating that the standard classification threshold is a modelling decision rather than an immutable property of the classifier.
However, it is not used to replace the main unweighted results because the purpose of the dissertation is primarily to compare the selected models and their explanations under a consistent evaluation framework.
The final methods are:
| Model                 | Final explanation method              |
| --------------------- | ------------------------------------- |
| Logistic Regression   | Signed coefficients                   |
| Decision Tree         | Feature importance and tree structure |
| Random Forest         | Feature importance                    |
| Multilayer Perceptron | Kernel SHAP                           |
LIME was examined during development but was not retained in the final dissertation analysis because preliminary local explanations were unstable or effectively zero for several diagnostic observations.
SHAP therefore became the final post-hoc explanation method.


### Results:
## 1. Logistic Regression Explanation Results

The final FICO-enriched Logistic Regression explanations use signed model coefficients.
The leading coefficients by absolute magnitude included:
| Rank | Feature                     | Coefficient |
| ---- | --------------------------- | ----------: |
| 1    | funded_amnt_inv             |     -0.5661 |
| 2    | term_months                 |     +0.4439 |
| 3    | installment                 |     +0.4034 |
| 4    | installment_to_income_ratio |     +0.3018 |
| 5    | fico_score                  |     -0.2764 |
| 6    | dti                         |     +0.2349 |

Positive coefficients increase the fitted log-odds of the default class, conditional on the other model terms.
Negative coefficients decrease the fitted log-odds of default.
The direction of important variables was generally intuitive.
For example:

Higher FICO score -> lower fitted default risk
Higher DTI        -> higher fitted default risk
Longer loan term  -> higher fitted default risk
Other important coefficients included recent inquiries, revolving utilisation, small-business loan purpose and rented home ownership.
The coefficient results should be interpreted as model associations rather than causal effects.

## 2. Decision Tree Explanation Results

The final FICO-enriched Decision Tree concentrated much of its importance in a relatively small number of variables.

Leading features were:

Rank	Feature	Importance
1	term_months	0.2611
2	fico_score	0.2541
3	installment_to_income_ratio	0.2139
4	dti	0.0887
5	mort_acc	0.0371
6	il_util	0.0332

The first four variables accounted for approximately:  81.78%
of the Decision Tree's recorded feature importance.
Other important variables included:

`tot_cur_bal`
`home_ownership`
`total_rev_hi_lim`
`all_util`

The Decision Tree therefore relied strongly on:

loan term;
credit quality;
borrower repayment burden; and
debt-to-income characteristics.

## 3. Random Forest Explanation Results

Random Forest distributed feature importance more broadly than the single Decision Tree.
Its leading FICO-enriched variables were:

| Rank | Feature                     | Importance |
| ---- | --------------------------- | ---------: |
| 1    | fico_score                  |     0.0869 |
| 2    | term_months                 |     0.0781 |
| 3    | loan_to_income_ratio        |     0.0664 |
| 4    | installment_to_income_ratio |     0.0619 |
| 5    | dti                         |     0.0580 |
| 6    | mort_acc                    |     0.0347 |


Other highly ranked features included:

revol_util
annual_inc
all_util
total_rev_hi_lim
tot_cur_bal
open_acc_to_total_acc_ratio

The Random Forest therefore showed a more distributed risk structure than the Decision Tree.

## 4. Final MLP SHAP Configuration
Kernel SHAP was applied to the final saved FICO-enriched Multilayer Perceptron.

The final configuration was:

MLP transformed features:       135

Background candidate pool:      500 observations

K-means background centres:     10

Global SHAP test observations:  100

Kernel SHAP nsamples:           200
The background expected default probability in the final run was:  0.146754
Kernel SHAP was used because the Spark MLP does not provide direct native explanations comparable with Logistic Regression coefficients or tree feature importance.
Because Kernel SHAP is computationally expensive, the final global analysis uses a representative sample rather than all 90,884 test observations.
The resulting SHAP analysis should therefore be interpreted as a sampled model-agnostic explanation of the final MLP.

## 4.1 MLP Global SHAP Results
The leading transformed features in the final MLP SHAP experiment were:
| Rank | Transformed Feature             | Mean Absolute SHAP |
| ---- | ------------------------------- | -----------------: |
| 1    | dti                             |            0.02241 |
| 2    | revol_util                      |            0.01808 |
| 3    | fico_score                      |            0.01598 |
| 4    | open_acc_to_total_acc_ratio     |            0.01562 |
| 5    | inq_last_6mths                  |            0.01199 |
| 6    | purpose_encoded_other           |            0.00745 |
| 7    | emp_length_years_missing        |            0.00566 |
| 8    | tot_coll_amt_missing            |            0.00495 |
| 9    | total_rev_hi_lim                |            0.00481 |
| 10   | home_ownership_encoded_MORTGAGE |            0.00469 |
This ranking evaluates each transformed input separately.
Consequently, one-hot encoded categorical variables can appear as multiple individual transformed features.

## 4.2 Grouped Global SHAP Results
To improve business interpretation, one-hot encoded components were also aggregated back to their original variables.
Signed component-level SHAP contributions belonging to the same original variable were first combined within an observation, after which mean absolute grouped importance was calculated.
The final grouped ranking was:
| Rank | Original Variable           | Grouped Mean Absolute SHAP |
| ---- | --------------------------- | -------------------------: |
| 1    | purpose                     |                    0.02432 |
| 2    | dti                         |                    0.02241 |
| 3    | revol_util                  |                    0.01808 |
| 4    | addr_state                  |                    0.01633 |
| 5    | fico_score                  |                    0.01598 |
| 6    | open_acc_to_total_acc_ratio |                    0.01562 |
| 7    | inq_last_6mths              |                    0.01199 |
| 8    | home_ownership              |                    0.00920 |
| 9    | emp_length_years_missing    |                    0.00566 |
| 10   | tot_coll_amt_missing        |                    0.00495 |
The grouped ranking differs from the transformed-feature ranking because categorical information can be distributed across multiple one-hot encoded inputs.
For example, the individual purpose categories are separated in the transformed input space but belong to the same original `purpose` variable.

## 4.3 Local SHAP Case Analysis

Four high-confidence test observations were selected for detailed local explanation.
The cases represent:
True Positive
True Negative
False Positive
False Negative

The final selected cases were:
| Case           | Actual Class | Predicted Class | Default Probability |
| -------------- | -----------: | --------------: | ------------------: |
| True Positive  |            1 |               1 |            0.905627 |
| True Negative  |            0 |               0 |            0.023276 |
| False Positive |            0 |               1 |            0.882493 |
| False Negative |            1 |               0 |            0.019843 |
The same SHAP expected value was used for all four:  0.146754

For each case:

Expected probability + sum of 135 SHAP contributions = MLP predicted default probability

The final numerical additivity check returned:
difference = 0.0

for all four diagnostic cases.

## 4.3.1 True-Positive SHAP Case
The selected true-positive loan:

Actual class:        Default
Predicted class:     Default
Default probability: 0.905627
Important upward contributions included:
60-month term                    +0.14240
Moving purpose                   +0.11759
Six delinquencies in two years   +0.08775
DTI = 32.02                      +0.06747
FICO score = 672                 +0.04866
Revolving utilisation = 85.9     +0.03971
The important features generally moved in the same direction and pushed predicted default probability substantially above the background probability.
This represents a high-confidence correct default prediction.

## 4.3.2 True-Negative SHAP Case

The selected true-negative loan:
Actual class:        Non-default
Predicted class:     Non-default
Default probability: 0.023276
Important downward contributions included:
FICO score = 822                  -0.02108
Total utilisation = 3.0          -0.01691
DTI = 0.35                        -0.01494
Revolving utilisation = 3.3      -0.01301
Open/total account ratio = 0.2727 -0.01017
Mortgage accounts = 7             -0.00881
These variables pushed the model substantially below the expected default probability.
The case therefore represents a high-confidence correct non-default prediction.

## 4.3.3 False-Positive SHAP Case

The selected false-positive loan:
Actual class:        Non-default
Predicted class:     Default
Default probability: 0.882493
Important upward contributions included:
Small-business purpose                     +0.13283
60-month term                              +0.12786
12 revolving accounts opened in 24 months +0.06972
Open/total account ratio = 1.0             +0.06633
Loan-to-income ratio = 0.4213              +0.06167
Installment-to-income ratio = 0.1666       +0.04874
FICO score = 672                           +0.04161
This case shows how several plausible risk indicators can combine to produce a confident incorrect prediction.
In a real lending context, this type of error could result in a creditworthy borrower being unnecessarily rejected or subjected to additional review.

## 4.3.4 False-Negative SHAP Case

The selected false-negative loan:
Actual class:        Default
Predicted class:     Non-default
Default probability: 0.019843
The largest downward contributions included:
Total revolving credit limit = 1,998,700  -0.13252
Total current balance = 3,437,283          -0.09067
Annual income = 400,000                    -0.01810
FICO score = 742                           -0.01219
Important upward contributions included:
Revolving balance / income = 4.3668 +0.04871
Florida residence                  +0.01940
Installment                        +0.01288
Three recent inquiries             +0.00754
The apparently protective balance, credit-limit, income and FICO characteristics dominated several risk-increasing signals.
The case demonstrates how a high-confidence false negative can occur when unusual financial values strongly affect the fitted model.
From a credit-risk perspective, this type of prediction is particularly important because the lender may fail to identify a borrower who subsequently defaults.

### Cross-Model XAI Comparison
The final explanation comparison combines:

Logistic Regression coefficients;
Decision Tree feature importance;
Random Forest feature importance; and
MLP global SHAP importance.

Because these methods use different numerical scales, the raw importance values are not compared directly.

Instead, features are compared using their rankings.

The final comparison uses:

Original-feature consensus
Top-10 Jaccard similarity
Top-20 Jaccard similarity
Spearman feature-rank correlation

### Cross-Model Consensus Features

Seven original variables had at least one Top-20 component in all four final explanation methods:

`fico_score`
`dti`
`open_acc_to_total_acc_ratio`
`revol_util`
`total_rev_hi_lim`
`mort_acc`
`home_ownership`
Their final best ranks were:
| Original Variable           | LR Rank | DT Rank | RF Rank | MLP Rank | Average Best Rank |
| --------------------------- | ------: | ------: | ------: | -------: | ----------------: |
| fico_score                  |       5 |       2 |       1 |        3 |              2.75 |
| dti                         |       6 |       4 |       5 |        1 |              4.00 |
| open_acc_to_total_acc_ratio |       9 |      11 |      12 |        4 |              9.00 |
| revol_util                  |      15 |      16 |       7 |        2 |             10.00 |
| total_rev_hi_lim            |      13 |       9 |      10 |        9 |             10.25 |
| mort_acc                    |      19 |       5 |       6 |       13 |             10.75 |
| home_ownership              |      20 |       8 |      19 |       10 |             14.25 |

FICO score therefore achieved the strongest overall cross-model consensus.
Debt-to-income ratio was the second strongest consensus variable.

Model artifacts, evaluation outputs and Delta tables are stored in Databricks, while code and final dissertation figures are retained in the GitHub repository.
