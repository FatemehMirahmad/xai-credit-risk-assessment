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
                                                          fico_score
                                                          fico_score_missing
                                                          pub_rec_bankruptcies
                                                          pub_rec_bankruptcies_missing
                                                          has_bankruptcy
                                                          mort_acc
                                                          mort_acc_missing

Final eligible observations:

- 455,318

## Target
Binary target variable: default_flag
- **Non-default** (Fully Paid) = 0
- **Default** (Default, Charged Off, Late (31-120 days), Late (16-30 days)) = 1

Final counts:

- Training observations: 364,434
- Test observations: 90,884

Training target distribution:

- Non-default: 271,018
- Default: 93,381

Test target distribution:

- Non-default: 67,389
- Default: 23,495

The complete dataset contained 338,411 non-default observations
(74.15%) and 116,866 default observations (25.85%).


## Final Models

### Logistic Regression

- `maxIter = 50`

### Decision Tree

- `maxDepth = 8`

### Random Forest

- `numTrees = 50`
- `maxDepth = 12`

### Multilayer Perceptron

- Architecture: `[135, 64, 32, 2]`
- `maxIter = 50`
- `blockSize = 256`
- `solver = l-bfgs`

