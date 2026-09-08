from pyspark.sql.functions import col, round as spark_round

lr_metrics = spark.table("dissertation.lendingclub.lc_2007_2017_logistic_regression_metrics")
dt_metrics = spark.table("dissertation.lendingclub.lc_2007_2017_decision_tree_metrics")
rf_metrics = spark.table("dissertation.lendingclub.lc_2007_2017_random_forest_metrics")
mlp_metrics = spark.table("dissertation.lendingclub.lc_2007_2017_deep_learning_mlp_metrics")

comparison_df = (
    lr_metrics
    .unionByName(dt_metrics)
    .unionByName(rf_metrics)
    .unionByName(mlp_metrics)
)

comparison_rounded = comparison_df.select(
    col("model"),
    spark_round(col("auc"), 4).alias("auc"),
    spark_round(col("accuracy"), 4).alias("accuracy"),
    spark_round(col("f1"), 4).alias("weighted_f1"),
    spark_round(col("precision"), 4).alias("weighted_precision"),
    spark_round(col("recall"), 4).alias("weighted_recall"),
    spark_round(col("default_precision"), 4).alias("default_precision"),
    spark_round(col("default_recall"), 4).alias("default_recall"),
    spark_round(col("default_f1"), 4).alias("default_f1")
)

display(comparison_rounded)

print("Models ranked by AUC:")
display(
    comparison_rounded.orderBy(col("auc").desc())
)

print("Models ranked by default recall:")
display(
    comparison_rounded.orderBy(col("default_recall").desc())
)

comparison_rounded.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable("dissertation.lendingclub.lc_2007_2017_model_comparison")

print("Saved table: dissertation.lendingclub.lc_2007_2017_model_comparison")


lr_fico_metrics = spark.table("dissertation.lendingclub.lc_2007_2017_logistic_regression_fico_metrics")
dt_fico_metrics = spark.table("dissertation.lendingclub.lc_2007_2017_decision_tree_fico_metrics")
rf_fico_metrics = spark.table("dissertation.lendingclub.lc_2007_2017_random_forest_fico_metrics")
mlp_fico_metrics = spark.table("dissertation.lendingclub.lc_2007_2017_deep_learning_mlp_fico_metrics")

comparison_fico_df = (
    lr_fico_metrics
    .unionByName(dt_fico_metrics)
    .unionByName(rf_fico_metrics)
    .unionByName(mlp_fico_metrics)
)

comparison_fico_rounded = comparison_fico_df.select(
    col("model"),
    spark_round(col("auc"), 4).alias("auc"),
    spark_round(col("accuracy"), 4).alias("accuracy"),
    spark_round(col("f1"), 4).alias("weighted_f1"),
    spark_round(col("precision"), 4).alias("weighted_precision"),
    spark_round(col("recall"), 4).alias("weighted_recall"),
    spark_round(col("default_precision"), 4).alias("default_precision"),
    spark_round(col("default_recall"), 4).alias("default_recall"),
    spark_round(col("default_f1"), 4).alias("default_f1")
)

display(comparison_fico_rounded)

print("Models FICO ranked by AUC:")
display(
    comparison_fico_rounded.orderBy(col("auc").desc())
)

print("Models FICO ranked by default recall:")
display(
    comparison_fico_rounded.orderBy(col("default_recall").desc())
)
comparison_fico_rounded.write \
    .mode("overwrite") \
    .format("delta") \
    .option("overwriteSchema", "true") \
    .saveAsTable(
        "dissertation.lendingclub.lc_2007_2017_model_comparison_fico"
    )

print(
    "Saved table: "
    "dissertation.lendingclub.lc_2007_2017_model_comparison_fico"
)