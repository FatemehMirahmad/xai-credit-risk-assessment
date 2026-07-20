raw_path = "/Volumes/dissertation/lendingclub/raw/lc_2016_2017.csv"
df_raw = (spark.read
.format('csv')
# 'header','true' means the first row is the header Without this option, Spark would create columns like: _c0, _c1, _c2, _c3
.option('header','true') 
# inferSchema: Set to "true" to automatically detect column data types (e.g., Integer, String). For the raw table, we want to preserve the original dataset as much as possible Then later, in lc_clean, we convert them (hence the bronze and silve idea). Bronze / lc_raw   = raw data, minimal changes. Silver / lc_clean = cleaned and typed data. Gold / lc_ml      = ML-ready data
.option('inferSchema','false')
# This tells Spark that quotation marks are used to wrap text values.
.option('quote','\"')
# This tells Spark how to handle escaped quotation marks inside quoted text. For example, if a text field contains a quote inside it, Spark knows how to interpret it.
.option('escape','\"')
.load(raw_path)
)

# It saves the DataFrame permanently as a Databricks table. 
# .mode('overwrite') means that if the table already exists, it will be overwritten. 
# .format('delta') means that the table will be stored in the Delta Lake format. 
# .saveAsTable('lendingclub.lc_raw')
df_raw.write.mode('overwrite').format('delta').saveAsTable('dissertation.lendingclub.lc_raw')