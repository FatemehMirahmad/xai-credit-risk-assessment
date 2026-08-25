from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col


df = spark.read.table('dissertation.lendingclub.lc_2007_2017_ml_no_leakage_fico')
print('Number of rows: ', df.count())
print('Number of columns: ',(len(df.columns)))

categorical_cols = [c for c, t in df.dtypes if  t == 'string' and c!= 'id']
numerical_cols = [c for c, t in df.dtypes if t in ['int','bigint','float','double'] and c not in  ['default_flag', 'id']]
print('Categorical columns: ', categorical_cols)
print('Numerical columns: ', numerical_cols)

train_df ,test_df = df.randomSplit([0.8,0.2], seed = 42)
# distribution of target variable in train and test set
print('Train set distribution: ')
display(train_df.groupBy('default_flag').count().orderBy('default_flag'))
print('Test set distribution: ')
display(test_df.groupBy('default_flag').count().orderBy('default_flag'))


Indexer = [
    StringIndexer(inputCol = c,
                  outputCol= f'{c}_indexed',
                  handleInvalid = 'keep'
                  )
           for c in categorical_cols
           ]
encoders = [OneHotEncoder(inputCol= f'{c}_indexed',
                          outputCol= f'{c}_encoded'
                          )
            for c in categorical_cols
            ]
# list of encoded columns
encoded_cols = [f'{c}_encoded' for c in categorical_cols]

assembler = VectorAssembler(inputCols = numerical_cols + encoded_cols,
                             outputCol= 'features')

rf = RandomForestClassifier(featuresCol= 'features',
                            labelCol= 'default_flag',
                            predictionCol= 'prediction',
                            probabilityCol= 'probability',
                            rawPredictionCol= 'rawPrediction',
                            numTrees = 50,
                            maxDepth = 12,
                            seed = 42
                            )
rf_pipeline = Pipeline(stages = Indexer + encoders + [assembler,rf])
rf_model = rf_pipeline.fit(train_df)
print('Random Forest model trained.')


rf_predictions = rf_model.transform(test_df).select('default_flag','prediction','probability','rawPrediction')
display(rf_predictions.limit(10))

rf_predictions.write.mode('overwrite').format('delta').saveAsTable('dissertation.lendingclub.lc_2007_2017_random_forest_fico_predictions')
# random forest evaluation

auc_evaluator = BinaryClassificationEvaluator(labelCol='default_flag',
                                    rawPredictionCol='rawPrediction',
                                    metricName='areaUnderROC'
                                    )
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='default_flag',
                                             predictionCol= 'prediction',
                                             metricName='accuracy'
                                             )
f1_evaluator = MulticlassClassificationEvaluator(labelCol='default_flag',
                                                predictionCol='prediction',
                                                metricName='f1'
                                            )

precision_evaluator = MulticlassClassificationEvaluator(labelCol='default_flag',
                                                        predictionCol='prediction',
                                                        metricName='weightedPrecision'
                                                    )

recall_evaluator = MulticlassClassificationEvaluator(labelCol='default_flag',
                                                    predictionCol='prediction',
                                                    metricName='weightedRecall'
                                                )
auc = auc_evaluator.evaluate(rf_predictions)
accuracy = accuracy_evaluator.evaluate(rf_predictions)
f1 = f1_evaluator.evaluate(rf_predictions)
precision = precision_evaluator.evaluate(rf_predictions)
recall = recall_evaluator.evaluate(rf_predictions)
confusion = (rf_predictions.groupBy('default_flag', 'prediction').count().orderBy('default_flag', 'prediction'))
print('Model: ','Random Forest')
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

print('Default Precision:', default_precision)
print('Default Recall:', default_recall)
print('Default F1:', default_f1)

metrics_rows = [('Random Forest FICO',float(auc),float(accuracy),float(f1),float(precision),float(recall),float(default_precision),float(default_recall),float(default_f1))]
metrics_schema = ['model','auc','accuracy','f1','precision','recall','default_precision','default_recall','default_f1']
rf_metrics = spark.createDataFrame(metrics_rows, metrics_schema)

display(rf_metrics)

rf_metrics.write.mode('overwrite') \
    .format('delta') \
    .option('overwriteSchema', 'true') \
    .saveAsTable('dissertation.lendingclub.lc_2007_2017_random_forest_fico_metrics')

print('Saved table: dissertation.lendingclub.lc_2007_2017_random_forest_fico_metrics')

# saving random forest model pipleline
spark.sql('''
CREATE VOLUME IF NOT EXISTS dissertation.lendingclub.model_artifacts
''')
MODEL_PATH = 'dbfs:/Volumes/dissertation/lendingclub/model_artifacts/lc_2007_2017_random_forest_fico_pipeline'
rf_model.write().overwrite().save(MODEL_PATH)

print('Saved Random Forest pipeline model to:', MODEL_PATH)
