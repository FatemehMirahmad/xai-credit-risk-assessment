import gc

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

gc.collect()
print("Cleared ML cache before starting.")

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
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



# Descision Tree
dt = DecisionTreeClassifier(
    featuresCol='features_unscaled',
    labelCol='default_flag',
    predictionCol='prediction',
    probabilityCol='probability',
    rawPredictionCol="rawPrediction",
    maxDepth = 8,
    seed = 42
)
dt_pipeline = Pipeline(stages = Indexer + encoders + [assembler, dt] )
dt_model = dt_pipeline.fit(train_df)
dt_predictions = (dt_model.transform(test_df).select('default_flag','prediction','probability','rawPrediction'))
display(dt_predictions.select('default_flag','prediction','probability','rawPrediction').limit(10))
# saving dt_prediction 
dt_predictions.write.mode('overwrite').format('delta').saveAsTable('dissertation.lendingclub.lc_2007_2017_decision_tree_fico_predictions')


# evaluation
auc_evaluator = BinaryClassificationEvaluator(
    labelCol = 'default_flag',
    rawPredictionCol  ='probability',
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


auc = auc_evaluator.evaluate(dt_predictions)
accuracy = accuracy_evaluator.evaluate(dt_predictions)
f1 = f1_evaluator.evaluate(dt_predictions)
precision = precision_evaluator.evaluate(dt_predictions)
recall = recall_evaluator.evaluate(dt_predictions)
confusion = (dt_predictions.groupBy('default_flag','prediction').count().orderBy('default_flag', "prediction"))

print('Model: ','Decision Tree')
print('AUC: ',auc)
print('Accuracy: ',accuracy)
print('F1: ',f1)
print('Precision: ',precision)
print('Recall: ',recall)

print('Confusion Matrix: ')
display(confusion)

#  Calculating metric for defaults only (not the weighted average of both classes classes)

tp = confusion.filter((col('default_flag') == 1) & (col('prediction')==1)).select('count').collect()[0][0]
fp = confusion.filter((col('default_flag') == 0) & (col('prediction') == 1)).select('count').collect()[0][0]
tn = confusion.filter((col('default_flag') == 0) & (col('prediction') == 0)).select('count').collect()[0][0]
fn = confusion.filter((col('default_flag') == 1) & (col('prediction') == 0)).select('count').collect()[0][0]

default_precision = tp/(tp+fp)
default_recall = tp/(tp+fn)
default_f1 = 2*(default_precision*default_recall)/(default_precision+default_recall)
print("Default class precision:", default_precision)
print("Default class recall:", default_recall)
print("Default class F1:", default_f1)

metrics_rows = [('Decision Tree FICO',float(auc),float(accuracy),float(f1),float(precision),float(recall),float(default_precision),float(default_recall),float(default_f1))]
metrics_schema = ['model','auc','accuracy','f1','precision','recall','default_precision','default_recall','default_f1']

metrics_df = spark.createDataFrame(metrics_rows, metrics_schema)
display(metrics_df)

metrics_df.write.mode("overwrite").option('overwriteSchema','True').format("delta").saveAsTable("dissertation.lendingclub.lc_2007_2017_decision_tree_fico_metrics")
print("Saved table: dissertation.lendingclub.lc_2007_2017_decision_tree_fico_metrics")

MODEL_PATH = "dbfs:/Volumes/dissertation/lendingclub/model_artifacts/lc_2007_2017_decision_tree_fico_pipeline"
dt_model.write().overwrite().save(MODEL_PATH)
print("Saved Decision Tree FICO pipeline model to:", MODEL_PATH)


