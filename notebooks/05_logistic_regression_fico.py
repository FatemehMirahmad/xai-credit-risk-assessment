import gc
import time

# Free Spark Connect ML cache at the start - delete any model objects from previous runs
try:
    del lr_model
except NameError:
    pass

try:
    del lr_pipeline
except NameError:
    pass

try:
    del lr_predictions
except NameError:
    pass

try:
    del dt_model
except NameError:
    pass

try:
    del dt_pipeline
except NameError:
    pass

try:
    del dt_predictions
except NameError:
    pass

# Force aggressive garbage collection to free Spark Connect ML cache
for _ in range(10):
    gc.collect()
time.sleep(15)  # Give server more time to process cache cleanup
print('Cleared ML cache before starting.')
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator , MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, lit

df = spark.table('dissertation.lendingclub.lc_2007_2017_ml_no_leakage_fico')
print('number of rows: ',df.count())
print('number of columns: ', len(df.columns))
display(df.limit(5))


# Defining categorical and numerical columns
categorical_cols = [c for c, t in df.dtypes if  t == 'string' and c!= 'id']
numerical_cols = [c for c, t in df.dtypes if t in ['int','bigint','float','double'] and c not in  ['default_flag', 'id']]
print('Categorical columns: ',categorical_cols)
print('Numerical columns: ',numerical_cols)

# Train and Test split
train_df, test_df = df.randomSplit([0.8,0.2], seed = 42)
print('number of train_df rows: ',train_df.count())
print('number of test_df rows: ',test_df.count())

# Indexing the categorical columns
Indexer = [ 
    StringIndexer(
        inputCol = c,
        outputCol = f'{c}_indexed',
        handleInvalid = 'keep'
    )
    for c in categorical_cols
]
# encoding the categorical columns
encoders = [ OneHotEncoder(
    inputCol = f'{c}_indexed',
    outputCol = f'{c}_encoded'
    )
    for c in categorical_cols
]

# creating a list of encoded columns
encoded_cols = [f'{c}_encoded' for c in categorical_cols]

# creating a vector of all the numerical and encoded columns by A feature transformer that merges multiple columns into a vector column.
assembler = VectorAssembler(
    inputCols = numerical_cols + encoded_cols,
    outputCol = 'features_unscaled',
    handleInvalid = 'keep')

# Creating a scaler to scale the features to their standard deviation
scaler = StandardScaler(
        inputCol = 'features_unscaled',
        outputCol = 'features',
        withStd = True,
        withMean = False
    )


# creating the Logistic Regression model
lr = LogisticRegression( 
                        featuresCol = 'features',
                        labelCol = 'default_flag',
                        predictionCol = 'prediction',
                        probabilityCol = 'probability',
                        maxIter = 50
)
lr_pipeline = Pipeline(stages = Indexer + encoders + [assembler, scaler, lr])

lr_model = lr_pipeline.fit(train_df)

# when you run .transform(), pyspark adds three columns to the dataframe: prediction, probability, and rawPrediction
lr_predictions = (lr_model.transform(test_df).select('default_flag','prediction','probability','rawPrediction'))
display(lr_predictions.select('default_flag','prediction','probability').limit(10))

# saving lr_predictions to delta table
lr_predictions.write.mode('overwrite').format('delta').saveAsTable('dissertation.lendingclub.lc_2007_2017_logistic_regression_fico_predictions')

auc_evaluator = BinaryClassificationEvaluator(
    labelCol = 'default_flag',
    rawPredictionCol  ='rawPrediction',
    metricName = 'areaUnderROC'
)
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol = 'default_flag',
    predictionCol = 'prediction',
    metricName = 'accuracy'
)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol = 'default_flag',
    predictionCol = 'prediction',
    metricName = 'f1'
)
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol = 'default_flag',
    predictionCol = 'prediction',
    metricName = 'weightedPrecision',
)
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol = 'default_flag',
    predictionCol = 'prediction',
    metricName = 'weightedRecall'
)

auc = auc_evaluator.evaluate(lr_predictions)
accuracy = accuracy_evaluator.evaluate(lr_predictions)
f1 = f1_evaluator.evaluate(lr_predictions)
precision = precision_evaluator.evaluate(lr_predictions)
recall = recall_evaluator.evaluate(lr_predictions)
confusion = (lr_predictions.groupBy('default_flag','prediction').count().orderBy('default_flag','prediction'))
print('Model: ','Logistic Regression')
print('AUC: ',auc)
print('Accuracy: ',accuracy)
print('F1: ',f1)
print('Precision: ',precision)
print('Recall: ',recall)

print('Confusion Matrix: ')
display(confusion)
cm_rows = confusion.collect()

cm = {
    (int(row['default_flag']), int(row['prediction'])): row['count']
    for row in cm_rows
}

tn = cm.get((0, 0), 0)
fp = cm.get((0, 1), 0)
fn = cm.get((1, 0), 0)
tp = cm.get((1, 1), 0)

default_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
default_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
default_f1 = (
    2 * default_precision * default_recall / (default_precision + default_recall)
    if (default_precision + default_recall) > 0
    else 0
)

print('Default class precision:', default_precision)
print('Default class recall:', default_recall)
print('Default class F1:', default_f1)

metrics_rows = [('Logistic Regression FICO',float(auc),float(accuracy),float(f1),float(precision),float(recall),float(default_precision),float(default_recall),float(default_f1))]
metrics_schema = ['model','auc','accuracy','f1','precision','recall','default_precision','default_recall','default_f1']
metrics_df = spark.createDataFrame(metrics_rows,metrics_schema)

display(metrics_df)

metrics_df.write.mode('overwrite').option('overwriteSchema','True').format('delta').saveAsTable('dissertation.lendingclub.lc_2007_2017_logistic_regression_fico_metrics')
print('Saved table: dissertation.lendingclub.lc_2007_2017_logistic_regression_fico_metrics')

spark.sql('''
CREATE VOLUME IF NOT EXISTS dissertation.lendingclub.model_artifacts
''')
MODEL_PATH = 'dbfs:/Volumes/dissertation/lendingclub/model_artifacts/lc_2007_2017_logistic_regression_fico_pipeline'
lr_model.write().overwrite().save(MODEL_PATH)
print('Saved Logistic Regression FICO pipeline model to: ',MODEL_PATH)
