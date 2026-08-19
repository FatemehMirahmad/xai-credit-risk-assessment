# lc_raw
raw_path = '/Volumes/dissertation/lendingclub/raw/lc_2016_2017.csv'
df_raw = (spark.read
          .format('csv')
          .option('header','true')
          .option('inferSchema','false')
          .option('quote','\"')
          .option('escape','\"')
          .option("multiLine", "true")
          .option("ignoreLeadingWhiteSpace", "true")
          .option("ignoreTrailingWhiteSpace", "true")
          .load(raw_path)
          )
df_raw.write.mode('overwrite').format('delta').saveAsTable('dissertation.lendingclub.lc_raw')
print('number of rows lc_raw: ', df_raw.count())
print('number of columns lc_raw: ', len(df_raw.columns))

# lc_loan
raw_path_lc_loan = '/Volumes/dissertation/lendingclub/raw/lc_loan.csv'
lc_loan = (spark.read
           .format('csv')
           .option('header','true')
           .option('inferSchema','false')
           .option('quote','\"')
           .option('escape','\"')
           .option("multiLine", "true")
           .option("ignoreLeadingWhiteSpace", "true")
           .option("ignoreTrailingWhiteSpace", "true")
           .load(raw_path_lc_loan)
           )
lc_loan.write.mode('overwrite').format('delta').saveAsTable('dissertation.lendingclub.lc_loan')
print('number of rows lc_loan: ', lc_loan.count())
print('number of columns lc_loan: ', len(lc_loan.columns))

# accepted 2007-2018
accepted_path = '/Volumes/dissertation/lendingclub/raw/accepted_2007_to_2018Q4.csv'
df_accepted = (spark.read
               .format('csv')
               .option('header','true') 
               .option('inferSchema','false')
               .option('quote','\"')
               .option('escape','\"')
               .option("multiLine", "true")
               .option("ignoreLeadingWhiteSpace", "true")
               .option("ignoreTrailingWhiteSpace", "true")
               .load(accepted_path)
               )
columns = [c for c in df_accepted.columns]
print(columns)
# display(df_accepted.limit(10))
print('number of rows accepted_2007_2018_raw: ', df_accepted.count())
print('number of columns accepted_2007_2018_raw: ', len(df_accepted.columns))
df_accepted.write.mode('overwrite').format('delta').saveAsTable('dissertation.lendingclub.accepted_2007_2018_raw')






