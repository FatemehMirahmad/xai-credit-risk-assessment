from pyspark.sql.functions import col, regexp_replace, regexp_extract ,trim , when, month, months_between, to_date, lit, concat, expr
import pandas as pd
df_raw = spark.table("dissertation.lendingclub.lc_2007_2017_raw_fico")
df_raw.select("loan_status").distinct().show()

def to_double(c):
    return expr(f"try_cast(`{c}` as double)")
def to_int(c):
    return expr(f"try_cast(try_cast(`{c}` as double) as int)")

df_clean = df_raw.select(
    col("id").cast("string").alias("id"),
    to_double("loan_amnt").alias("loan_amnt"),
    to_double("funded_amnt").alias("funded_amnt"),
    to_double("funded_amnt_inv").alias("funded_amnt_inv"),

    trim(col("term")).alias("term"),

    expr("try_cast(regexp_replace(`int_rate`, '%', '') as double)").alias("int_rate"),

    to_double("installment").alias("installment"),

    trim(col("grade")).alias("grade"),
    trim(col("sub_grade")).alias("sub_grade"),

    trim(col("emp_length")).alias("emp_length"),
    trim(col("home_ownership")).alias("home_ownership"),
    to_double("annual_inc").alias("annual_inc"),
    trim(col("verification_status")).alias("verification_status"),
    trim(col("earliest_cr_line")).alias("earliest_cr_line"),


    trim(col("issue_d")).alias("issue_d"),
    trim(col("loan_status")).alias("loan_status"),
    trim(col("purpose")).alias("purpose"),
    trim(col("addr_state")).alias("addr_state"),

    to_double("dti").alias("dti"),
    to_int("delinq_2yrs").alias("delinq_2yrs"),
    to_int("inq_last_6mths").alias("inq_last_6mths"),
    to_double("mths_since_last_delinq").alias("mths_since_last_delinq"),
    to_int("open_acc").alias("open_acc"),
    to_int("pub_rec").alias("pub_rec"),
    to_double("revol_bal").alias("revol_bal"),
    expr("try_cast(regexp_replace(`revol_util`, '%', '') as double)").alias("revol_util"),

    
    to_int("total_acc").alias("total_acc"),
    trim(col("initial_list_status")).alias("initial_list_status"),
    to_int("collections_12_mths_ex_med").alias("collections_12_mths_ex_med"),
    to_double("mths_since_last_major_derog").alias("mths_since_last_major_derog"),
    trim(col("application_type")).alias("application_type"),
    to_int("acc_now_delinq").alias("acc_now_delinq"),
    to_double("tot_coll_amt").alias("tot_coll_amt"),
    to_double("tot_cur_bal").alias("tot_cur_bal"),
    to_double("total_rev_hi_lim").alias("total_rev_hi_lim"),

    # extra credit variables
    to_int("open_acc_6m").alias("open_acc_6m"),
    to_int("open_il_12m").alias("open_il_12m"),
    to_int("open_il_24m").alias("open_il_24m"),
    to_double("mths_since_rcnt_il").alias("mths_since_rcnt_il"),
    to_double("total_bal_il").alias("total_bal_il"),
    to_double("il_util").alias("il_util"),
    to_int("open_rv_12m").alias("open_rv_12m"),
    to_int("open_rv_24m").alias("open_rv_24m"),
    to_double("max_bal_bc").alias("max_bal_bc"),
    to_double("all_util").alias("all_util"),
    to_int("inq_fi").alias("inq_fi"),
    to_int("total_cu_tl").alias("total_cu_tl"),
    to_int("inq_last_12m").alias("inq_last_12m"),

    to_double("fico_score").alias("fico_score"),
    to_double("pub_rec_bankruptcies").alias("pub_rec_bankruptcies"),
    to_double("mort_acc").alias("mort_acc"),

    col("source_period").cast("string").alias("source_period"),

    # repayment / post-loan variables
    to_double("total_pymnt").alias("total_pymnt"),
    to_double("recoveries").alias("recoveries"),
    to_double("last_pymnt_amnt").alias("last_pymnt_amnt")

    )
# removing unresolved loan status: Current, In Grace Period
df_clean = df_clean.filter(col("loan_status").isin("Fully Paid","Charged Off","Default","Late (31-120 days)","Late (16-30 days)"))
print("checking if loan_status is filtered: ")
df_clean.select("loan_status").distinct().show()

# Creating Default Flag
df_clean = df_clean.withColumn("default_flag",(when(col("loan_status").isin("Default","Charged Off","Late (31-120 days)","Late (16-30 days)"),1).otherwise(0)))
df_clean.select("default_flag").distinct().show()

# making emp_length_years out of emp_length
df_clean = df_clean.withColumn(
    "emp_length_years",
    when(col("emp_length").isNull(), lit(None).cast("int"))
    .when(trim(col("emp_length")) == "", lit(None).cast("int"))
    .when(trim(col("emp_length")) == "10+ years", lit(10))
    .when(trim(col("emp_length")) == "< 1 year", lit(0))
    .otherwise(
        expr("try_cast(regexp_extract(trim(emp_length), '(\\\\d+)', 1) as int)")
    )
)
# display(df_clean.select("emp_length_years").dtypes)

# turning term from 36 months to 36 and 60 months to 60 
df_clean = df_clean.withColumn("term_months",
    regexp_extract(trim(col("term")), r"(\d+)", 1).cast("int"))


df_clean = df_clean.withColumn(
    "credit_history_months",
    months_between(to_date(concat(lit("01-"),col("issue_d")),"dd-MMM-yyyy"),to_date(concat(lit("01-"),col("earliest_cr_line")),"dd-MMM-yyyy"))
    )

df_clean = df_clean.withColumn(
    "credit_history_years",
    col("credit_history_months") / lit(12)
)
display(spark.sql("SELECT * FROM dissertation.lendingclub.lc_2007_2017_raw_fico LIMIT 10"))
display(df_clean.limit(10))

essential_cols = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "term_months",
    "installment",
    "loan_status",
    "default_flag"
]
print("Clean rows before removing nulls:", df_clean.count())
for c in essential_cols:
    df_clean = df_clean.filter(col(c).isNotNull())
print("Clean rows after removing nulls:", df_clean.count())
df_clean = df_clean.filter(col("loan_amnt").cast("double") > 0)
df_clean = df_clean.filter(col("annual_inc").cast("double") >= 0)
df_clean = df_clean.filter(col("dti").cast("double") >= 0)
print("Clean rows after filter:", df_clean.count())
display(df_clean.filter(col("dti").cast("double") < 0))
print("Clean rows:", df_clean.count())
print("Clean columns:", len(df_clean.columns))

display(
    df_clean.groupBy("loan_status", "default_flag")
            .count()
            .orderBy("default_flag", "loan_status")
)

display(df_clean.limit(10))

# Saving the df_clean
df_clean.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable("dissertation.lendingclub.lc_2007_2017_clean")
print("Saved Table: dissertation.lendingclub.lc_2007_2017_clean")