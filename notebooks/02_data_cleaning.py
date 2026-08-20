from pyspark.sql.functions import col, regexp_replace, regexp_extract ,trim , when, month, months_between, to_date, lit, concat
import pandas as pd
df_raw = spark.table('dissertation.lendingclub.lc_raw')
df_raw.select('loan_status').distinct().show()
df_clean = df_raw.select(
    col('id').cast('string').alias('id'),
    col('loan_amnt').cast('double').alias('loan_amnt'),
    col('funded_amnt').cast('double').alias('funded_amnt'),
    col('funded_amnt_inv').cast('double').alias('funded_amnt_inv'),

    col('term').alias('term'),

    regexp_replace(col('int_rate').cast('string'), '%', '').cast('double').alias('int_rate'),

    col('installment').cast('double').alias('installment'),

    trim(col('grade')).alias('grade'),
    trim(col('sub_grade')).alias('sub_grade'),

    trim(col('emp_length')).alias('emp_length'),
    trim(col('home_ownership')).alias('home_ownership'),
    col('annual_inc').cast('double').alias('annual_inc'),
    trim(col('verification_status')).alias('verification_status'),
    trim(col('earliest_cr_line')).alias('earliest_cr_line'),


    trim(col('issue_d')).alias('issue_d'),
    col('loan_status').alias('loan_status'),
    trim(col('purpose')).alias('purpose'),
    trim(col('addr_state')).alias('addr_state'),
    col('dti').cast('double').alias('dti'),
    col('delinq_2yrs').cast('int').alias('delinq_2yrs'),
    col('inq_last_6mths').cast('int').alias('inq_last_6mths'),
    col('mths_since_last_delinq').cast('double').alias('mths_since_last_delinq'),
    col('open_acc').cast('int').alias('open_acc'),
    col('pub_rec').cast('int').alias('pub_rec'),
    col('revol_bal').cast('double').alias('revol_bal'),
    regexp_replace(col('revol_util').cast('string'), '%', '').cast('double').alias('revol_util'),

    
    col('total_acc').cast('int').alias('total_acc'),
    trim(col('initial_list_status')).alias('initial_list_status'),
    col('collections_12_mths_ex_med').cast('int').alias('collections_12_mths_ex_med'),
    col('mths_since_last_major_derog').cast('int').alias('mths_since_last_major_derog'),
    trim(col('application_type')).alias('application_type'),
    col('acc_now_delinq').cast('int').alias('acc_now_delinq'),
    col('tot_coll_amt').cast('double').alias('tot_coll_amt'),
    col('tot_cur_bal').cast('double').alias('tot_cur_bal'),
    col('total_rev_hi_lim').cast('double').alias('total_rev_hi_lim'),

    # extra credit variables
    col('open_acc_6m').cast('int').alias('open_acc_6m'),
    col('open_il_12m').cast('int').alias('open_il_12m'),
    col('open_il_24m').cast('int').alias('open_il_24m'),
    col('mths_since_rcnt_il').cast('double').alias('mths_since_rcnt_il'),
    col('total_bal_il').cast('double').alias('total_bal_il'),
    col('il_util').cast('double').alias('il_util'),
    col('open_rv_12m').cast('int').alias('open_rv_12m'),
    col('open_rv_24m').cast('int').alias('open_rv_24m'),
    col('max_bal_bc').cast('double').alias('max_bal_bc'),
    col('all_util').cast('double').alias('all_util'),
    col('inq_fi').cast('int').alias('inq_fi'),
    col('total_cu_tl').cast('int').alias('total_cu_tl'),
    col('inq_last_12m').cast('int').alias('inq_last_12m'),

    # repayment / post-loan variables
    col('total_pymnt').cast('double').alias('total_pymnt'),
    col('recoveries').cast('double').alias('recoveries'),
    col('last_pymnt_amnt').cast('double').alias('last_pymnt_amnt')

    )
# removing unresolved loan status: Current, In Grace Period
df_clean = df_clean.filter(col('loan_status').isin('Fully Paid','Charged Off','Default','Late (31-120 days)','Late (16-30 days)'))
print('checking if loan_status is filtered: ')
df_clean.select('loan_status').distinct().show()

# Creating Default Flag
df_clean = df_clean.withColumn('default_flag',(when(col('loan_status').isin('Default','Charged Off','Late (31-120 days)','Late (16-30 days)'),1).otherwise(0)))
df_clean.select('default_flag').distinct().show()

# making emp_length_years out of emp_length
df_clean = df_clean.withColumn('emp_length_years',when(col('emp_length').isNull(), lit(None).cast('int'))
        .when(trim(col('emp_length')) == '10+ years', lit(10))
        .when(trim(col('emp_length')) == '< 1 year', lit(0))
        .otherwise(regexp_extract(trim(col('emp_length')), r'(\d+)', 1).cast('int')))
# display(df_clean.select('emp_length_years').dtypes)

# turning term from 36 months to 36 and 60 months to 60 
df_clean = df_clean.withColumn('term_months',
    regexp_extract(trim(col('term')), r'(\d+)', 1).cast('int'))


df_clean = df_clean.withColumn(
    'credit_history_months',
    months_between(to_date(concat(lit('01-'),col('issue_d')),'dd-MMM-yyyy'),to_date(concat(lit('01-'),col('earliest_cr_line')),'dd-MMM-yyyy'))
    )

df_clean = df_clean.withColumn(
    'credit_history_years',
    col('credit_history_months') / lit(12)
)
display(spark.sql('SELECT * FROM dissertation.lendingclub.lc_raw LIMIT 10'))
display(df_clean.limit(10))

essential_cols = [
    'loan_amnt',
    'annual_inc',
    'dti',
    'term_months',
    'installment',
    'loan_status',
    'default_flag'
]
print('Clean rows before removing nulls:', df_clean.count())
for c in essential_cols:
    df_clean = df_clean.filter(col(c).isNotNull())
print('Clean rows after removing nulls:', df_clean.count())
df_clean = df_clean.filter(col('loan_amnt').cast('double') > 0)
df_clean = df_clean.filter(col('annual_inc').cast('double') >= 0)
df_clean = df_clean.filter(col('dti').cast('double') >= 0)
print('Clean rows after filter:', df_clean.count())
display(df_clean.filter(col('dti').cast('double') < 0))
print('Clean rows:', df_clean.count())
print('Clean columns:', len(df_clean.columns))

display(
    df_clean.groupBy('loan_status', 'default_flag')
            .count()
            .orderBy('default_flag', 'loan_status')
)

display(df_clean.limit(10))

# Saving the df_clean
df_clean.write.mode('overwrite').option('overwriteSchema', 'true').format('delta').saveAsTable('dissertation.lendingclub.lc_clean')
print('Saved Table: dissertation.lendingclub.lc_clean')