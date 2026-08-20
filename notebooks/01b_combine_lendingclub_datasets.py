from pyspark.sql.functions import count, lit
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

df_2007_2015 = df_2007_2015.withColumn('source_period', lit(2007_2015))
df_2016_2017 = df_2016_2017.withColumn('source_period', lit(2016_2017))

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