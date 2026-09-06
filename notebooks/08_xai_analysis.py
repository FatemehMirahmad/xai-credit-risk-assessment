from pyspark.sql.functions import col, desc, count, row_number
from pyspark.ml import PipelineModel
from pyspark.sql.window import Window
df = spark.table('dissertation.lendingclub.lc_2007_2017_ml_no_leakage_fico')
print('Number of rows: ',df.count())
print('Number of columns: ',len(df.columns))

LR_MODEL_PATH = (
    'dbfs:/Volumes/dissertation/lendingclub/model_artifacts/lc_2007_2017_logistic_regression_fico_pipeline')
DT_MODEL_PATH = (
    'dbfs:/Volumes/dissertation/lendingclub/model_artifacts/lc_2007_2017_decision_tree_fico_pipeline')
RF_MODEL_PATH = (
    'dbfs:/Volumes/dissertation/lendingclub/model_artifacts/lc_2007_2017_random_forest_fico_pipeline')
lr_pipeline = PipelineModel.load(LR_MODEL_PATH)
dt_pipeline = PipelineModel.load(DT_MODEL_PATH)
rf_pipeline = PipelineModel.load(RF_MODEL_PATH)
print('Saved models loaded.')

def get_feature_names(pipeline_model, source_df):
    sample = pipeline_model.transform(
        source_df.limit(1)
    )
    final_model = pipeline_model.stages[-1]
    vector_col = final_model.getFeaturesCol()
    metadata = sample.schema[vector_col].metadata
    attrs = (
        metadata
        .get('ml_attr', {})
        .get('attrs', {})
    )
    feature_info = []
    for attr_type in ['numeric', 'binary', 'nominal']:
        feature_info.extend(
            attrs.get(attr_type, [])
        )
    feature_info = sorted(
        feature_info,
        key=lambda x: x['idx']
    )
    feature_names = [
        x['name']
        for x in feature_info
    ]

    # Logistic Regression uses scaled 'features'.
    # If scaling removed the names, use features_unscaled metadata.
    if (
        len(feature_names) != final_model.numFeatures
        and 'features_unscaled' in sample.columns
    ):
        metadata = sample.schema['features_unscaled'].metadata
        attrs = (
            metadata
            .get('ml_attr', {})
            .get('attrs', {})
        )
        feature_info = []

        for attr_type in ['numeric', 'binary', 'nominal']:
            feature_info.extend(
                attrs.get(attr_type, [])
            )
        feature_info = sorted(
            feature_info,
            key=lambda x: x['idx']
        )
        feature_names = [
            x['name']
            for x in feature_info
        ]
    return feature_names    

lr_model = lr_pipeline.stages[-1]
lr_feature_names = get_feature_names(
    lr_pipeline,df)
lr_coefficients = lr_model.coefficients.toArray()
print('Number of LR features:',len(lr_feature_names))
print('Number of LR coefficients:',len(lr_coefficients))
lr_explanation_rows = [
    (
        feature,
        float(coefficient),
        float(abs(coefficient))
    )
    for feature, coefficient
    in zip(lr_feature_names, lr_coefficients)
]

lr_explanation = spark.createDataFrame(
    lr_explanation_rows,
    ['feature','coefficient','absolute_coefficient'])
print("Top Logistic Regression features by absolute coefficient:")

display(
    lr_explanation
    .orderBy(desc("absolute_coefficient"))
    .limit(20)
)
print("Features increasing predicted default risk:")

display(
    lr_explanation
    .filter(col("coefficient") > 0)
    .orderBy(desc("coefficient"))
    .limit(15)
)
print("Features decreasing predicted default risk:")

display(
    lr_explanation
    .filter(col("coefficient") < 0)
    .orderBy(col("coefficient"))
    .limit(15)
)
lr_explanation.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_logistic_regression_fico_coefficients"
    )

print(
    "Saved table: "
    "dissertation.lendingclub."
    "lc_xai_logistic_regression_fico_coefficients"
)
# ---------------------------------------------------------
# Decision Tree - Global Explanation
# ---------------------------------------------------------

dt_model = dt_pipeline.stages[-1]

dt_feature_names = get_feature_names(
    dt_pipeline,
    df
)

dt_importances = (
    dt_model
    .featureImportances
    .toArray()
)
dt_explanation_rows = [
    (
        feature,
        float(importance)
    )
    for feature, importance
    in zip(dt_feature_names, dt_importances)
]

dt_explanation = spark.createDataFrame(
    dt_explanation_rows,
    [
        "feature",
        "importance"
    ]
)
print("Top Decision Tree features:")

display(
    dt_explanation
    .orderBy(desc("importance"))
    .limit(20)
)
print("Decision Tree structure:")
print(dt_model.toDebugString)
dt_explanation.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_decision_tree_fico_importance"
    )

print(
    "Saved table: "
    "dissertation.lendingclub."
    "lc_xai_decision_tree_fico_importance"
)
# ---------------------------------------------------------
# Random Forest - Global Explanation
# ---------------------------------------------------------

rf_model = rf_pipeline.stages[-1]

rf_feature_names = get_feature_names(
    rf_pipeline,
    df
)

rf_importances = (
    rf_model
    .featureImportances
    .toArray()
)
rf_explanation_rows = [
    (
        feature,
        float(importance)
    )
    for feature, importance
    in zip(rf_feature_names, rf_importances)
]

rf_explanation = spark.createDataFrame(
    rf_explanation_rows,
    [
        "feature",
        "importance"
    ]
)
print("Top Random Forest features:")

display(
    rf_explanation
    .orderBy(desc("importance"))
    .limit(20)
)
rf_explanation.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_random_forest_fico_importance"
    )

print(
    "Saved table: "
    "dissertation.lendingclub."
    "lc_xai_random_forest_fico_importance"
)
interpretability_rows = [
    (
        "Logistic Regression",
        "Coefficients",
        "Yes",
        "Yes",
        "High"
    ),
    (
        "Decision Tree",
        "Feature importance and decision rules",
        "Yes",
        "Yes",
        "High"
    ),
    (
        "Random Forest",
        "Aggregated feature importance",
        "Yes",
        "Limited",
        "Medium"
    ),
    (
        "Multilayer Perceptron",
        "No simple native feature explanation",
        "No",
        "No",
        "Low"
    )
]

interpretability_schema = [
    "model",
    "native_explanation",
    "global_interpretability",
    "local_interpretability",
    "interpretability_level"
]

interpretability_comparison = spark.createDataFrame(
    interpretability_rows,
    interpretability_schema
)

display(interpretability_comparison)
# cross-model importance comparison
lr_ranked = (
    lr_explanation
    .withColumn(
        "lr_rank",
        row_number().over(
            Window.orderBy(
                col("absolute_coefficient").desc()
            )
        )
    )
    .select(
        "feature",
        col("absolute_coefficient").alias("lr_importance"),
        col("coefficient").alias("lr_coefficient"),
        "lr_rank"
    )
)
dt_ranked = (
    dt_explanation
    .withColumn(
        "dt_rank",
        row_number().over(
            Window.orderBy(
                col("importance").desc()
            )
        )
    )
    .select(
        "feature",
        col("importance").alias("dt_importance"),
        "dt_rank"
    )
)
rf_ranked = (
    rf_explanation
    .withColumn(
        "rf_rank",
        row_number().over(
            Window.orderBy(
                col("importance").desc()
            )
        )
    )
    .select(
        "feature",
        col("importance").alias("rf_importance"),
        "rf_rank"
    )
)
cross_model_xai = (
    lr_ranked
    .join(
        dt_ranked,
        on="feature",
        how="outer"
    )
    .join(
        rf_ranked,
        on="feature",
        how="outer"
    )
)
cross_model_xai = cross_model_xai.withColumn(
    "average_rank",
    (
        col("lr_rank") +
        col("dt_rank") +
        col("rf_rank")
    ) / 3
)
print("Most consistently important features across models:")

display(
    cross_model_xai
    .filter(
        col("lr_rank").isNotNull() &
        col("dt_rank").isNotNull() &
        col("rf_rank").isNotNull()
    )
    .orderBy("average_rank")
    .limit(20)
)
cross_model_xai.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_cross_model_fico_comparison"
    )