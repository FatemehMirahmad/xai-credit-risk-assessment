
from pyspark.sql.functions import col,regexp_replace,trim

df_raw = spark.table('dissertation.lendingclub.lc_raw')
df_clean = df_raw.select(
    col('loan_amnt').cast('double').alias('loan_amnt'),
    col('funded_amnt').cast('double').alias('funded_amnt'),
    col('funded_amnt_inv').cast('double').alias('funded_amnt_inv'),

    trim(col('term')).alias('term'),
    regexp_replace(col('int_rate'),"%",'').cast('double').alias('int_rate'),
    col('installment').cast('double').alias('installment'),

    trim(col('emp_length')).alias('emp_length'),
    trim(col('home_ownership')).alias('home_ownership'),
    col('annual_inc').cast('double').alias('annual_inc'),
    trim(col('verification_status')).alias('verification_status'),
    trim(col('loan_status')).alias('loan_status'),
    trim(col('purpose')).alias('purpose'),
    trim(col('addr_state')).alias('addr_state'),
    
    col('dti').cast('double').alias('dti'),
    # credit-history count variables
    col('delinq_2yrs').cast('int').alias('delinq_2yrs'),
    col('inq_last_6mths').cast('int').alias('inq_last_6mths'),
    col('total_acc').cast('int').alias('total_acc'),
    col('open_acc').cast('int').alias('open_acc'),
    col('pub_rec').cast('int').alias('pub_rec'),

    col('revol_bal').cast('double').alias('revol_bal'),
    regexp_replace(col('revol_util'),'%','').cast('double').alias('revol_util'),
    col('collections_12_mths_ex_med').cast('int').alias('collections_12_mths_ex_med'),

    col("mths_since_last_major_derog").cast("int").alias("mths_since_last_major_derog"),
    trim(col("application_type")).alias('application_type'),
    col("acc_now_delinq").cast("int").alias("acc_now_delinq"),
    col("tot_coll_amt").cast("double").alias("tot_coll_amt"),
    col("tot_cur_bal").cast("double").alias("tot_cur_bal"),
    col("total_rev_hi_lim").cast("double").alias("total_rev_hi_lim"),
    col("inq_last_12m").cast("int").alias("inq_last_12m"),
    # repayment/future variables kept only for analysis, not ML
    col("total_pymnt").cast("double").alias("total_pymnt"),
    col("recoveries").cast("double").alias("recoveries"),
    col("last_pymnt_amnt").cast("double").alias("last_pymnt_amnt")
    )
df_clean = df_clean.filter(
    (col("loan_amnt").isNotNull()) &
    (col("annual_inc").isNotNull()) &
    (col("loan_status").isNotNull())
)
df_clean.write.mode('overwrite').format('delta').saveAsTable('dissertation.lendingclub.lc_clean')

display(spark.sql('DESCRIBE DETAIL dissertation.lendingclub.lc_clean'))