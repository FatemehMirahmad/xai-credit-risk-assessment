# Final Data Dictionary

## XAI Credit-Risk Assessment

This document describes the variables retained in the final modelling datasets used in the dissertation.

Two modelling conditions are used:

- **No-FICO condition:** base LendingClub application and credit-profile variables.
- **FICO-enriched condition:** the same base variables plus FICO, bankruptcy and mortgage-account information.

The enriched dataset should therefore be described as **FICO/credit-profile enriched**, not as a FICO-only dataset.

The final FICO-enriched modelling dataset contains 64 columns, including the binary target variable.

---

## 1. Target Variable

| Feature | Definition | Dataset | Construction |
|---|---|---|---|
| `default_flag` | Binary credit-outcome target | Both | 1 = Charged Off, Default, Late (31–120 days), or Late (16–30 days); 0 = Fully Paid |

Loan statuses that had not reached a sufficiently clear outcome, including `Current` and `In Grace Period`, were excluded before modelling.

---

## 2. Categorical Variables

| Feature | Definition | Dataset | Treatment |
|---|---|---|---|
| `home_ownership` | Borrower's home-ownership status | Both | One-hot encoded |
| `verification_status` | Income-verification status | Both | One-hot encoded |
| `purpose` | Borrower-reported purpose of the loan | Both | One-hot encoded |
| `addr_state` | US state supplied in the application | Both | One-hot encoded |
| `initial_list_status` | Initial LendingClub listing status | Both | One-hot encoded |
| `application_type` | Individual or joint application | Both | Standardised and one-hot encoded |

Application-type values were standardised so that inconsistent labels such as `Individual` and `INDIVIDUAL` were represented consistently as `INDIVIDUAL`, while `Joint App` and `JOINT` were represented as `JOINT`.

---

## 3. Original Numerical Variables

| Feature | Definition | Dataset |
|---|---|---|
| `loan_amnt` | Loan amount requested by the borrower | Both |
| `funded_amnt` | Total amount funded for the loan | Both |
| `funded_amnt_inv` | Amount funded by investors | Both |
| `installment` | Scheduled monthly loan payment | Both |
| `annual_inc` | Borrower's self-reported annual income | Both |
| `dti` | Debt-to-income ratio | Both |
| `delinq_2yrs` | Number of 30+ day delinquencies in the previous two years | Both |
| `inq_last_6mths` | Credit inquiries during the previous six months | Both |
| `mths_since_last_delinq` | Months since most recent delinquency | Both |
| `open_acc` | Number of open credit lines | Both |
| `pub_rec` | Number of derogatory public records | Both |
| `revol_bal` | Total revolving credit balance | Both |
| `revol_util` | Revolving credit utilisation rate | Both |
| `total_acc` | Total number of credit lines | Both |
| `collections_12_mths_ex_med` | Collections in previous 12 months excluding medical collections | Both |
| `acc_now_delinq` | Number of accounts currently delinquent | Both |
| `tot_coll_amt` | Total collection amount ever owed | Both |
| `tot_cur_bal` | Total current balance across accounts | Both |
| `open_acc_6m` | Open trades created during the previous six months | Both |
| `open_il_12m` | Installment accounts opened during the previous 12 months | Both |
| `open_il_24m` | Installment accounts opened during the previous 24 months | Both |
| `mths_since_rcnt_il` | Months since most recent installment account was opened | Both |
| `total_bal_il` | Total balance of installment accounts | Both |
| `il_util` | Installment-account utilisation | Both |
| `open_rv_12m` | Revolving trades opened during the previous 12 months | Both |
| `open_rv_24m` | Revolving trades opened during the previous 24 months | Both |
| `max_bal_bc` | Maximum current balance on revolving accounts | Both |
| `all_util` | Overall balance-to-credit-limit utilisation | Both |
| `total_rev_hi_lim` | Total revolving high-credit / credit limit | Both |
| `inq_fi` | Number of personal-finance inquiries | Both |
| `total_cu_tl` | Number of finance trades | Both |
| `inq_last_12m` | Number of credit inquiries during the previous 12 months | Both |

---

## 4. Engineered Numerical Variables

| Feature | Definition | Formula / Construction |
|---|---|---|
| `term_months` | Numeric loan term | Numeric value extracted from `term`, normally 36 or 60 |
| `emp_length_years` | Employment length in years | Parsed from `emp_length`; `< 1 year` = 0 and `10+ years` = 10 |
| `credit_history_months` | Length of credit history in months | `months_between(issue_d, earliest_cr_line)` |
| `credit_history_years` | Length of credit history in years | `credit_history_months / 12` |
| `loan_to_income_ratio` | Loan amount relative to annual income | `loan_amnt / annual_inc` |
| `installment_to_income_ratio` | Monthly installment relative to monthly income | `installment / (annual_inc / 12)` |
| `revol_bal_to_income_ratio` | Revolving balance relative to annual income | `revol_bal / annual_inc` |
| `open_acc_to_total_acc_ratio` | Proportion of total accounts currently open | `open_acc / total_acc` |

---

## 5. Engineered Risk Indicators

| Feature | Definition | Construction |
|---|---|---|
| `has_delinquency_2yrs` | Indicates recent delinquency | 1 if `delinq_2yrs > 0`, otherwise 0 |
| `has_public_record` | Indicates a derogatory public record | 1 if `pub_rec > 0`, otherwise 0 |
| `has_collections_12_mths` | Indicates recent collection activity | 1 if `collections_12_mths_ex_med > 0`, otherwise 0 |
| `has_mths_since_last_delinq` | Indicates whether previous-delinquency timing was available | 1 if original `mths_since_last_delinq` was present, otherwise 0 |

---

## 6. Base-Dataset Missingness Indicators

| Feature | Meaning |
|---|---|
| `emp_length_years_missing` | Employment-length information was missing before imputation |
| `mths_since_rcnt_il_missing` | Months-since-recent-installment information was missing |
| `il_util_missing` | Installment-utilisation information was missing |
| `tot_coll_amt_missing` | Total collection amount was missing |
| `tot_cur_bal_missing` | Total current balance was missing |
| `total_rev_hi_lim_missing` | Total revolving credit limit was missing |

Each flag equals 1 when the corresponding value was missing before imputation and 0 otherwise.

---

## 7. FICO/Credit-Profile Enrichment

The enriched modelling condition adds seven fields.

| Feature | Definition | Construction |
|---|---|---|
| `fico_score` | Borrower's origination FICO score | `(fico_range_low + fico_range_high) / 2` |
| `pub_rec_bankruptcies` | Number of public-record bankruptcies | LendingClub enrichment field |
| `mort_acc` | Number of mortgage accounts | LendingClub enrichment field |
| `has_bankruptcy` | Indicates whether a bankruptcy was recorded | 1 if `pub_rec_bankruptcies > 0`, otherwise 0 |
| `fico_score_missing` | FICO score missing before imputation | Binary missingness flag |
| `pub_rec_bankruptcies_missing` | Bankruptcy count missing before imputation | Binary missingness flag |
| `mort_acc_missing` | Mortgage-account count missing before imputation | Binary missingness flag |

The enriched condition therefore represents more than the addition of FICO score alone.

---

## 8. Missing-Value Treatment

Missing-value treatment depended on the meaning of the variable.

For selected variables where missingness had a meaningful interpretation, `-1` was used to preserve the absence of information.

Examples include:

- `mths_since_last_delinq`
- `emp_length_years`
- `mths_since_rcnt_il`
- `il_util`

Separate missingness flags were retained where appropriate.

Other numerical variables were filled using approximate median values calculated in the feature-engineering pipeline.

Categorical missing values were handled using a consistent placeholder where required.

Missing-value treatment was completed before model training so that the final ML tables did not contain unresolved null values in the selected predictors.

---

## 9. Raw Fields Used Only for Derivation

The following variables were required during data preparation but were not retained as final predictors.

| Raw Feature | Use |
|---|---|
| `loan_status` | Used to construct `default_flag` |
| `term` | Used to construct `term_months` |
| `emp_length` | Used to construct `emp_length_years` |
| `earliest_cr_line` | Used to calculate credit-history length |
| `issue_d` | Used with `earliest_cr_line` to calculate credit-history length |
| `fico_range_low` | Used to calculate `fico_score` |
| `fico_range_high` | Used to calculate `fico_score` |

---

## 10. Intentionally Excluded Variables

### LendingClub risk/pricing variables

The following variables were intentionally excluded:

- `grade`
- `sub_grade`
- `int_rate`

They reflect LendingClub's own risk assessment and pricing process and were treated as lender-derived risk information that could weaken the independence of the model comparison.

### Post-loan variables

Variables containing information generated after loan origination were also excluded from predictive modelling.

Examples include:

- `total_pymnt`
- `recoveries`
- `last_pymnt_amnt`

They were not used because they contain information about repayment behaviour that would not be available when an initial credit decision is made.

### Identifier

`id` was used for data alignment and enrichment but was not used as a model feature.

---

## 11. Final Dataset Dimensions

Final observations:

`455,318`

Final FICO-enriched ML-table columns:

`64`

Final FICO-enriched MLP transformed features after indexing and encoding:

`135`

The no-FICO and FICO-enriched datasets contain the same eligible borrower observations. The difference between them is the additional credit-profile information available to the enriched models.
