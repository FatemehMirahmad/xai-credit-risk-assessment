from pyspark.sql.functions import count, lit, col
df_2007_2015 = spark.table('dissertation.lendingclub.lc_loan')
df_2016_2017 = spark.table('dissertation.lendingclub.lc_raw')

print('2007–2015 rows:', df_2007_2015.count())
print('2016–2017 rows:', df_2016_2017.count())
print('2007–2015 columns:', len(df_2007_2015.columns))
print('2016–2017 columns:', len(df_2016_2017.columns))

# checking for difference in columns
cols_2007_2015 = set(df_2007_2015.columns)
cols_2016_2017 = set(df_2016_2017.columns)
print('column differences:', cols_2007_2015 - cols_2016_2017)

# dropping the two extra columns:
extra_cols = ['open_il_6m','url']
for c in extra_cols:
    if c in df_2007_2015.columns:
        df_2007_2015 = df_2007_2015.drop(c)

cols_2007_2015 = set(df_2007_2015.columns)
cols_2016_2017 = set(df_2016_2017.columns)
print('column differences:', cols_2007_2015 - cols_2016_2017)
print(cols_2007_2015)
print(cols_2016_2017)

df_2007_2015 = df_2007_2015.withColumn('source_period', lit('2007_2015'))
df_2016_2017 = df_2016_2017.withColumn('source_period', lit('2016_2017'))

df = df_2007_2015.unionByName(df_2016_2017)

print('union rows:', df.count())
print('union columns:', len(df.columns))

# checking for duplicate rows:
duplicate_ids = df.groupBy('id').agg(count('*').alias('row_count')).filter('row_count > 1')
print('Duplicate IDs:', duplicate_ids.count())
print('Duplicated ids: ')
display(duplicate_ids.limit(10))

# checking loan status distribution:
display(df.groupBy('loan_status').count().orderBy('loan_status'))

# saving
df.write.mode('overwrite') \
    .format('delta') \
    .option('overwriteSchema', 'true') \
    .saveAsTable('dissertation.lendingclub.lc_2007_2017_raw')

print('Saved table: dissertation.lendingclub.lc_2007_2017_raw')
# attaching FICO
df_accepted = spark.table("dissertation.lendingclub.accepted_2007_2018_raw")
credit_lookup = df_accepted.select(
    col('id').cast('string').alias('id'),
    col("fico_range_low").cast("double").alias("fico_range_low"),
    col("fico_range_high").cast("double").alias("fico_range_high"),
    col("pub_rec_bankruptcies").cast("double").alias("pub_rec_bankruptcies"),
    col("mort_acc").cast("double").alias("mort_acc")
    ).withColumn(
    "fico_score",
    (col("fico_range_low") + col("fico_range_high")) / lit(2)
    ).dropDuplicates(["id"])

df_combined_fico = (
    df.withColumn("id", col("id").cast("string"))
    .join(credit_lookup, on="id", how="left")
)

print("Rows after FICO join:", df_combined_fico.count())

display(
    df_combined_fico.select(
        "id",
        "source_period",
        "loan_status",
        "fico_range_low",
        "fico_range_high",
        "fico_score",
        "pub_rec_bankruptcies",
        "mort_acc"
    ).limit(10)
)
from pyspark.sql.functions import sum, when

display(
    df_combined_fico.select(
        sum(when(col("fico_score").isNull(), 1).otherwise(0)).alias("missing_fico_score"),
        sum(when(col("pub_rec_bankruptcies").isNull(), 1).otherwise(0)).alias("missing_pub_rec_bankruptcies"),
        sum(when(col("mort_acc").isNull(), 1).otherwise(0)).alias("missing_mort_acc")
    )
)
# saving combined table with fico
df_combined_fico.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable("dissertation.lendingclub.lc_2007_2017_raw_fico")

print("Saved table: dissertation.lendingclub.lc_2007_2017_raw_fico")
