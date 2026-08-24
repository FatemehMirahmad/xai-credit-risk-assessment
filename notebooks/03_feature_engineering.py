from pyspark.sql.functions import when, col, try_divide, lit, sum

df_labeled = spark.table('dissertation.lendingclub.lc_2007_2017_clean')

print('Input table: dissertation.lendingclub.lc_2007_2017_clean')
print('Rows:', df_labeled.count())
print('Columns:', len(df_labeled.columns))

display(df_labeled.limit(5))

print('Loan status and target distribution:')
display(
    df_labeled.groupBy('loan_status', 'default_flag')
              .count()
              .orderBy('default_flag', 'loan_status')
)

if 'source_period' in df_labeled.columns:
    print('Source period distribution:')
    display(
        df_labeled.groupBy('source_period')
                  .count()
                  .orderBy('source_period')
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
if 'pub_rec_bankruptcies' in df_labeled.columns:
    df_labeled = df_labeled.withColumn(
        'has_bankruptcy',
        when(col('pub_rec_bankruptcies') > 0, 1).otherwise(0)
    )


def prepare_ml_table(source_df,selected_cols,source_df_name):
    print(f'Preparing table: {source_df_name}')

    df_ml = source_df.select([c for c in selected_cols if c in source_df.columns])
    categorical_cols = [c for c,t in df_ml.dtypes if t == 'string']

    # checking null values
    print(f'Missing values before preparing {source_df_name} table:')
    missing_exprs = [
        sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df_ml.columns
    ]
    display(df_ml.select(missing_exprs))

    # creating missing flags:
    missing_columns = [
    'emp_length_years',
    'mths_since_rcnt_il',
    'il_util',
    'tot_coll_amt',
    'tot_cur_bal',
    'total_rev_hi_lim',
    'fico_score',
    'pub_rec_bankruptcies',
    'mort_acc']
    for c in missing_columns:
        if c in df_ml.columns:
            df_ml = df_ml.withColumn(
                f'{c}_missing',
                when(col(c).isNull(), 1).otherwise(0))


    # For emp_length_years, mths_since_rcnt_il, and il_util, -1 means 'missing information'
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


    # filling null values in catagorical columns with 'Unknown'
    df_ml = df_ml.fillna('Unknown', subset = categorical_cols)


    median_fill_cols = [
    'inq_last_6mths',
    'revol_util',
    'collections_12_mths_ex_med',
    'tot_coll_amt',
    'tot_cur_bal',
    'total_rev_hi_lim',

    'open_acc_6m',
    'open_il_12m',
    'open_il_24m',
    'total_bal_il',
    'open_rv_12m',
    'open_rv_24m',
    'max_bal_bc',
    'all_util',
    'inq_fi',
    'total_cu_tl',
    'inq_last_12m',

    'loan_to_income_ratio',
    'installment_to_income_ratio',
    'revol_bal_to_income_ratio',
    'open_acc_to_total_acc_ratio',

    'credit_history_months',
    'credit_history_years',

    'fico_score',
    'pub_rec_bankruptcies',
    'mort_acc'
]

    median_values = {}
    # for c in median_fill_cols:
    #     if c in df_ml.columns:
    #         median = df_ml.approxQuantile(c, [0.5], 0.01)[0]
    #         if median is not None:
    #             median_values[c] = median
    for c in median_fill_cols:
        if c in df_ml.columns:
            median_result = df_ml.approxQuantile(c, [0.5], 0.01)

            if len(median_result) > 0 and median_result[0] is not None:
                median_values[c] = float(median_result[0])
    # filling null values in numerical columns with median
    df_ml = df_ml.fillna(median_values)


    print(f'Missing values after preparing {source_df_name} table: ')
    missing_exprs = [
        sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df_ml.columns
    ]
    display(df_ml.select(missing_exprs))
    
    # Checking number of rows and columns
    print(f'Number of rows in {source_df_name}: ', df_ml.count())
    print(f'Number of columns in {source_df_name}: ', len(df_ml.columns))
    return df_ml

# selecting columns to keep
base_selected_cols= ['loan_amnt',
    'funded_amnt',
    'funded_amnt_inv',
    'term_months',
    'installment',
    'annual_inc',
    'dti',
    'delinq_2yrs',
    'inq_last_6mths',
    'mths_since_last_delinq',
    'open_acc',
    'pub_rec',
    'revol_bal',
    'revol_util',
    'total_acc',
    'collections_12_mths_ex_med',
    'acc_now_delinq',
    'tot_coll_amt',
    'tot_cur_bal',
    'open_acc_6m',
    'open_il_12m',
    'open_il_24m',
    'mths_since_rcnt_il',
    'total_bal_il',
    'il_util',
    'open_rv_12m',
    'open_rv_24m',
    'max_bal_bc',
    'all_util',
    'total_rev_hi_lim',
    'inq_fi',
    'total_cu_tl',
    'inq_last_12m',
    'emp_length_years',
    'credit_history_months',
    'credit_history_years',
    'loan_to_income_ratio',
    'installment_to_income_ratio',
    'revol_bal_to_income_ratio',
    'open_acc_to_total_acc_ratio',
    'has_delinquency_2yrs',
    'has_public_record',
    'has_collections_12_mths',
    'has_mths_since_last_delinq',
    'default_flag',
    'home_ownership',
    'verification_status',
    'purpose',
    'addr_state',
    'initial_list_status',
    'application_type']

print('Preparing df_ml_no_leakage: ')
df_ml_no_leakage = prepare_ml_table(df_labeled,base_selected_cols,'df_ml_no_leakage')
print('No-leakage table: ')
print('Rows:', df_ml_no_leakage.count())
print('Columns:', len(df_ml_no_leakage.columns))

# checking default flag distribution
print('Default flag distribution in df_ml_no_leakage: ')
display(
    df_ml_no_leakage.groupBy('default_flag')
        .count()
        .orderBy('default_flag')
        )


# saving the df_ml datafram as lc_ml table
df_ml_no_leakage.write.mode('overwrite').format('delta').option('overwriteSchema', 'true').saveAsTable('dissertation.lendingclub.lc_2007_2017_ml_no_leakage')
print('Saved table: dissertation.lendingclub.lc_2007_2017_ml_no_leakage')

# creating a new table with extra credit columns from the other source
extra_credit_cols  = ['fico_score',
    'pub_rec_bankruptcies',
    'mort_acc',
    'has_bankruptcy']

print('Preparing df_ml_no_leakage_fico: ')
df_ml_no_leakage_fico = prepare_ml_table(df_labeled,base_selected_cols + extra_credit_cols,'df_ml_no_leakage_fico')

print('No-leakage table with FICO / bankruptcy / mortgage features')
print('Rows:', df_ml_no_leakage_fico.count())
print('Columns:', len(df_ml_no_leakage_fico.columns))

# checking default flag distribution
print('Default flag distribution in df_ml_no_leakage_fico: ')
display(
    df_ml_no_leakage_fico.groupBy('default_flag')
        .count()
        .orderBy('default_flag')
        )
df_ml_no_leakage_fico.write.mode('overwrite').format('delta').option('overwriteSchema', 'true').saveAsTable('dissertation.lendingclub.lc_2007_2017_ml_no_leakage_fico')
print('Saved table: dissertation.lendingclub.lc_2007_2017_ml_no_leakage_fico')
