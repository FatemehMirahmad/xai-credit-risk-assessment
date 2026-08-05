from pyspark.sql.functions import col, round as spark_round

lr_metrics = spark.table("dissertation.lendingclub.lc_logistic_regression_metrics")
dt_metrics = spark.table("dissertation.lendingclub.lc_decision_tree_metrics")
rf_metrics = spark.table("dissertation.lendingclub.lc_random_forest_metrics")
mlp_metrics = spark.table("dissertation.lendingclub.lc_deep_learning_mlp_metrics")

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
    spark_round(col("f1"), 4).alias("f1"),
    spark_round(col("precision"), 4).alias("precision"),
    spark_round(col("recall"), 4).alias("recall"),
    spark_round(col("default_precision"), 4).alias("default_precision"),
    spark_round(col("default_recall"), 4).alias("default_recall"),
    spark_round(col("default_f1"), 4).alias("default_f1")
)

display(comparison_rounded)

comparison_rounded.write.mode("overwrite") \
    .format("delta") \
    .option("overwriteSchema", "true") \
    .saveAsTable("dissertation.lendingclub.lc_model_comparison")

print("Saved table: dissertation.lendingclub.lc_model_comparison")