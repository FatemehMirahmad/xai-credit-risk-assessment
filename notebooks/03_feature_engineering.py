from pyspark.sql.functions import when
from pyspark.sql.functions import col

df = spark.table('dissertation.lendingclub.lc_clean')
df_labeled = df.filter(col('loan_status').isin('Fully Paid','Charged Off','Default','Late (31-120 days)','Late (16-30 days)'))
# In PySpark, withColumn() is a DataFrame method used to add a new column or replace an existing column with the same name
df_labeled = df_labeled.withColumn('default_flag',when(col('loan_status').isin('Charged Off','Default','Late (31-120 days)','Late (16-30 days)'),1).otherwise(0))

df_labeled.select('loan_status').distinct().show()
# dataframe without leakage (excluding: grade/subgrade/int_rate)
df_ml_no_leakage = df_labeled.select('loan_amnt','funded_amnt','funded_amnt_inv','term','installment','emp_length','home_ownership','annual_inc','verification_status','purpose','addr_state','dti','delinq_2yrs','inq_last_6mths','total_acc','open_acc','pub_rec','revol_bal','revol_util','collections_12_mths_ex_med','acc_now_delinq','tot_coll_amt','tot_cur_bal','total_rev_hi_lim','inq_last_12m','default_flag')

df_ml_no_leakage.write.mode('overwrite').format('delta').saveAsTable('dissertation.lendingclub.lc_ml_no_leakage')

from pyspark.sql.functions import lit, try_divide
# Adding ratios
df_eng = spark.table('dissertation.lendingclub.lc_ml_no_leakage')
# It shows how large the loan is compared with the borrower’s yearly income.
# loan_income_ratio = how large the loan is compared with income
# installment_income_ratio = monthly repayment burden
# revol_bal_to_limit = revolving credit usage compared with credit limit
df_eng = df_eng.withColumn('loan_income_ratio',try_divide(col('loan_amnt'),col('annual_inc')))
# It shows how large the loan is compared with the borrower’s monthly
df_eng = df_eng.withColumn('installment_income_ratio',try_divide(col('installment'),col('annual_inc'))/lit(12))
df_eng = df_eng.withColumn('revol_bal_to_limit',try_divide(col('revol_bal'),col('total_rev_hi_lim')))
df_eng.write.mode('overwrite').format('delta').saveAsTable('dissertation.lendingclub.lc_ml_engineered')