from pyspark.sql.functions import when, col, try_divide, lit, count, sum

df_labeled = spark.table('dissertation.lendingclub.lc_clean')
display(
    df_labeled.groupBy("loan_status", "default_flag")
              .count()
              .orderBy("default_flag", "loan_status")
)

# creating ratios
df_labeled = df_labeled.withColumn('loan_to_income_ratio',try_divide(col('loan_amnt'),col('annual_inc')))
df_labeled = df_labeled.withColumn('installment_to_income_ratio',try_divide(col('installment'),col('annual_inc')/lit(12)))
df_labeled = df_labeled.withColumn('revol_bal_to_income_ratio',try_divide(col('revol_bal'),col('annual_inc')))
df_labeled = df_labeled.withColumn('open_acc_to_total_acc_ratio',try_divide(col('open_acc'),col('total_acc')))

# Creating simple risk flags
df_labeled = df_labeled.withColumn('has_delinquency_2yrs',when(col('delinq_2yrs')> 0, 1).otherwise(0))
df_labeled = df_labeled.withColumn('has_public_record',when(col('pub_rec')> 0, 1).otherwise(0))
df_labeled = df_labeled.withColumn('has_collections_12_mths',when(col('collections_12_mths_ex_med')> 0, 1).otherwise(0))
df_labeled = df_labeled.withColumn('has_mths_since_last_delinq',when(col('mths_since_last_delinq').isNotNull(), 1).otherwise(0))

# selecting columns to keep
selected_col= ["loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term_months",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "open_acc_6m",
    "open_il_12m",
    "open_il_24m",
    "mths_since_rcnt_il",
    "total_bal_il",
    "il_util",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "all_util",
    "total_rev_hi_lim",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
    "emp_length_years",
    "credit_history_months",
    "credit_history_years",
    "loan_to_income_ratio",
    "installment_to_income_ratio",
    "revol_bal_to_income_ratio",
    "open_acc_to_total_acc_ratio",
    "has_delinquency_2yrs",
    "has_public_record",
    "has_collections_12_mths",
    "has_mths_since_last_delinq",
    'default_flag',
    'home_ownership',
    'verification_status',
    'purpose',
    'addr_state',
    'initial_list_status',
    'application_type']


# Creating df_ml dataframe
df_ml = df_labeled.select([c for c in selected_col if c in df_labeled.columns])
display(df_ml.limit(10))
# seperating numerical and categorical columns
numerical_cols = [c for c,t in df_ml.dtypes if t in ['double','int','bigint', 'float'] and c != 'default_flag']
categorical_cols = [c for c,t in df_ml.dtypes if t == 'string']

# checking null values
missing_exprs = [
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df_ml.columns
]
display(df_ml.select(missing_exprs))

# columns with missing values: inq_last_6mths, revol_util, open_acc_6m, open_il_12m, open_il_24m, total_bal_il , open_rv_12m, open_rv_24m,  max_bal_bc, all_util, inq_fi, total_cu_tl, inq_last_12m, mths_since_last_delinq, emp_length_years, mths_since_rcnt_il, il_util

# creating missing flags:
missing_columns = ['emp_length_years', 'mths_since_rcnt_il', 'il_util']
for c in missing_columns:
    df_ml = df_ml.withColumn(f'{c}_missing', when(col(c).isNull(), 1).otherwise(0))


# For emp_length_years, mths_since_rcnt_il, and il_util, -1 means "missing information"
# already have 'has_mths_since_last_delinq' so mths_since_last_delinq flag does not need to be created
# filling missing values with -1
if 'mths_since_last_delinq' in df_ml.columns:
    df_ml = df_ml.fillna({'mths_since_last_delinq': -1})
if 'emp_length_years' in df_ml.columns:
    df_ml = df_ml.fillna({'emp_length_years': -1})
if 'mths_since_rcnt_il' in df_ml.columns:
    df_ml = df_ml.fillna({'mths_since_rcnt_il': -1})
if 'il_util' in df_ml.columns:
    df_ml = df_ml.fillna({'il_util': -1})

# display(df_ml.select('mths_since_last_delinq').distinct().orderBy('mths_since_last_delinq'))

# filling null values in catagorical columns with 'Unknown'
df_ml = df_ml.fillna("Unknown", subset = categorical_cols)


# calculate median for inq_last_6mths, revol_util, open_acc_6m, open_il_12m, open_il_24m, total_bal_il , open_rv_12m, open_rv_24m,  max_bal_bc, all_util, inq_fi, total_cu_tl, inq_last_12m columns
median_fill_cols = ['inq_last_6mths', 'revol_util', 'open_acc_6m', 'open_il_12m', 'open_il_24m', 'total_bal_il' , 'open_rv_12m', 'open_rv_24m',  'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m']
median_values = {}
for c in median_fill_cols:
    if c in df_ml.columns:
        median = df_ml.approxQuantile(c, [0.5], 0.01)[0]
        if median is not None:
            median_values[c] = median
# filling null values in numerical columns with median
df_ml = df_ml.fillna(median_values)

# Checking number of rows and columns
print('Number of rows: ', df_ml.count())
print('Number of columns: ', len(df_ml.columns))

# checking default flag distribution
display(
    df_ml.groupBy('default_flag')
        .count()
        .orderBy('default_flag')
        )

missing_exprs = [
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df_ml.columns
]
display(df_ml.select(missing_exprs))

# saving the df_ml datafram as lc_ml table
df_ml.write.mode('overwrite').format('delta').option('overwriteSchema', 'true').saveAsTable('dissertation.lendingclub.lc_ml_no_leakage')

