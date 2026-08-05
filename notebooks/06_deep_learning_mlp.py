from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

import gc
objects_to_delete = [
    "lr_model",
    "lr_pipeline",
    "lr_predictions",
    "dt_model",
    "dt_pipeline",
    "dt_predictions",
    "rf_model",
    "rf_pipeline",
    "rf_predictions",
    "preprocessing_model",
    "preprocessing_pipeline",
    "mlp",
    "mlp_model",
    "mlp_predictions"
]

for object_name in objects_to_delete:
    if object_name in globals():
        del globals()[object_name]

gc.collect()

print("Cleared previous ML objects.")

df = spark.table('dissertation.lendingclub.lc_ml_no_leakage')
print('Number of rows: ', df.count())
print('Number of columns: ',(len(df.columns)))

# checking distribution of target variable
print('Distribution of target variable: ')
display(
    df.groupBy('default_flag')
      .count()
      .orderBy('default_flag')
)
categorical_cols = [c for c, t in df.dtypes if t == 'string']
numerical_cols = [c for c, t in df.dtypes if t in ['double','int','bigint','float'] and c != 'default_flag']
print('Categorical columns: ', categorical_cols)
print('Numerical columns: ', numerical_cols)

train_df ,test_df = df.randomSplit([0.8,0.2], seed = 42)
# distribution of target variable in train and test set
print('Distribution of target variable in train set: ')
display(train_df.groupBy('default_flag').count().orderBy('default_flag'))

print('Distribution of target variable in test set: ')
display(test_df.groupBy('default_flag').count().orderBy('default_flag'))

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

preprocessing_pipeline = Pipeline(stages = Indexer + encoders + [assembler, scaler])
preprocessing_model = preprocessing_pipeline.fit(train_df)
print("Preprocessing pipeline fitted.")

train_prepared = preprocessing_model.transform(train_df)
test_prepared = preprocessing_model.transform(test_df)
print("Prepared train and test data.")

print("Prepared training data:")
display(train_prepared.limit(5))

# first faeture row
first_feature_row = train_prepared.select('features').first()
print('First feature row: ', first_feature_row)

if first_feature_row is None:
    raise ValueError('The prepared training dataset is emtpy.')

input_size = first_feature_row['features'].size
print('Number of input features: ', input_size)

# Neural Network Architecture
# nember of neurons in each hidden layer (2 comes from the number of classes)
layers = [input_size,64 , 32, 2]
print('MLP architecture: ', layers)

# Creating Multilayer Perceptron Classifier
mlp = MultilayerPerceptronClassifier(featuresCol='features', labelCol='default_flag', predictionCol='prediction',rawPredictionCol= 'rawPrediction', layers=layers, maxIter=50, blockSize=256, solver= 'l-bfgs', seed=42)
print("MLP configuration:")
print("Layers:", layers)
print("Maximum iterations:", mlp.getMaxIter())
print("Block size:", mlp.getBlockSize())
print("Solver:", mlp.getSolver())

# training the model
mlp_model = mlp.fit(train_prepared)
print("Multilayer Perceptron training completed.")

# predicting on test data
mlp_predictions = mlp_model.transform(test_prepared).select('default_flag', 'prediction','probability','rawPrediction')
print("Predictions made.")
print("Number of predictions:", mlp_predictions.count())
display(mlp_predictions.limit(10))

# evaluating the model
auc = BinaryClassificationEvaluator(labelCol='default_flag',
                                    rawPredictionCol='rawPrediction',
                                    metricName='areaUnderROC'
                                    )
accuracy = MulticlassClassificationEvaluator(labelCol='default_flag',
                                             predictionCol= 'prediction',
                                             metricName='accuracy'
                                             )
f1_evaluator = MulticlassClassificationEvaluator(labelCol='default_flag',
                                                predictionCol="prediction",
                                                metricName="f1"
                                            )

precision_evaluator = MulticlassClassificationEvaluator(labelCol='default_flag',
                                                        predictionCol="prediction",
                                                        metricName="weightedPrecision"
                                                    )

recall_evaluator = MulticlassClassificationEvaluator(labelCol='default_flag',
                                                    predictionCol="prediction",
                                                    metricName="weightedRecall"
                                                )
auc = auc.evaluate(mlp_predictions)
accuracy = accuracy.evaluate(mlp_predictions)
f1 = f1_evaluator.evaluate(mlp_predictions)
precision = precision_evaluator.evaluate(mlp_predictions)
recall = recall_evaluator.evaluate(mlp_predictions)
confusion = (mlp_predictions.groupBy('default_flag', "prediction").count().orderBy('default_flag', "prediction"))
print('Model: ','MLP')
print('AUC: ',auc)
print('Accuracy: ',accuracy)
print('F1: ',f1)
print('Precision: ',precision)
print('Recall: ',recall)

print('Confusion Matrix: ')
display(confusion)

cm_rows = confusion.collect()

cm = {
    (int(row["default_flag"]), int(row["prediction"])): row["count"]
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

print("Default Precision:", default_precision)
print("Default Recall:", default_recall)
print("Default F1:", default_f1)

metrics_rows = [
    (
        "Multilayer Perceptron",
        float(auc),
        float(accuracy),
        float(f1),
        float(precision),
        float(recall),
        float(default_precision),
        float(default_recall),
        float(default_f1)
    )
]
metrics_schema = metrics_schema = [
    "model",
    "auc",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "default_precision",
    "default_recall",
    "default_f1"
]
mlp_metrics = spark.createDataFrame(metrics_rows, metrics_schema)

display(mlp_metrics)
# saving mlp metrics


mlp_metrics.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable('dissertation.lendingclub.lc_deep_learning_mlp_metrics')

print("Saved table: dissertation.lendingclub.lc_deep_learning_mlp_metrics")

# saving mlp model and preprocessing model
spark.sql("""
CREATE VOLUME IF NOT EXISTS
dissertation.lendingclub.model_artifacts
""")

preprocessing_model.write().overwrite().save("dbfs:/Volumes/dissertation/lendingclub/model_artifacts/deep_learning_mlp_preprocessing")
mlp_model.write().overwrite().save("dbfs:/Volumes/dissertation/lendingclub/model_artifacts/deep_learning_mlp_model")
mlp_predictions.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable("dissertation.lendingclub.lc_deep_learning_mlp_predictions")

print("Saved table: dissertation.lendingclub.lc_deep_learning_mlp_predictions")
print("Saved preprocessing model to: dbfs:/Volumes/dissertation/lendingclub/model_artifacts/deep_learning_mlp_preprocessing")
print("Saved MLP model to: dbfs:/Volumes/dissertation/lendingclub/model_artifacts/deep_learning_mlp_model")
print("06_deep_learning_mlp.py completed successfully.")