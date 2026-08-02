from pyspark.sql.functions import round as spark_round,col, when, lit
lr_metrics_table = spark.read.table("dissertation.lendingclub.lc_logistic_regression_metrics")
dt_metrics_table = spark.read.table("dissertation.lendingclub.lc_decision_tree_metrics")
display(lr_metrics_table)
display(dt_metrics_table)

# combining the metrics tables
metrics_df = lr_metrics_table.unionByName(dt_metrics_table)
display(metrics_df)

# rounding the metrics
comparison_df = metrics_df.select(col('model'),
                                  spark_round(col('auc'),4).alias('auc'),
                                  spark_round(col('accuracy'),4).alias('accuracy'),
                                  spark_round(col('f1'),4).alias('f1'),
                                  spark_round(col('precision'),4).alias('precision'),
                                  spark_round(col('recall'),4).alias('recall'),
                                  spark_round(col('default_precision'),4).alias('default_precision'),
                                  spark_round(col('default_recall'),4).alias('default_recall'),
                                  spark_round(col('default_f1'),4).alias('default_f1')
                                  )
display(comparison_df)


print("Best model by AUC")
display(comparison_df.orderBy(col('auc').desc()).limit(1))

print('Best model by Default Recall')
display(comparison_df.orderBy(col('default_recall').desc()).limit(1))

comparison_final = comparison_df.withColumn('interpretation', when(
        col("model") == "Logistic Regression",
        lit("Better baseline model so far; stronger AUC and better default recall than Decision Tree.")
    ).when(
        col("model") == "Decision Tree",
        lit("More interpretable tree-based baseline, but weaker default detection in current results.")
    ).otherwise(
        lit("Baseline model.")
    ))

display(comparison_final)

comparison_final.write.mode('overwrite').option('overwriteSchema', 'true').saveAsTable("dissertation.lendingclub.lc_baseline_comparison")
print('Baseline comparison table is saved to dissertation.lendingclub.lc_baseline_comparison')