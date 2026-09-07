# 1. Imports

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import Window

from pyspark.sql.functions import (
    col,
    lit,
    desc,
    asc,
    sum as spark_sum,
    min as spark_min,
    first,
    row_number,
    when
)


# 2. Configuration

TOP_K_VALUES = [
    10,
    20
]


TOP_FEATURES_TO_DISPLAY = 20


OUTPUT_DIR = (
    "/Volumes/dissertation/"
    "lendingclub/xai_outputs"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print(
    "XAI evaluation output directory:",
    OUTPUT_DIR
)


# 3. Load final global XAI tables

lr_xai = spark.table(
    "dissertation.lendingclub."
    "lc_xai_logistic_regression_fico_coefficients"
)


dt_xai = spark.table(
    "dissertation.lendingclub."
    "lc_xai_decision_tree_fico_importance"
)


rf_xai = spark.table(
    "dissertation.lendingclub."
    "lc_xai_random_forest_fico_importance"
)


mlp_xai = spark.table(
    "dissertation.lendingclub."
    "lc_xai_mlp_fico_shap_global"
)


print(
    "Final XAI tables loaded."
)


# 4. Inspect source table sizes

print(
    "LR features:",
    lr_xai.count()
)

print(
    "DT features:",
    dt_xai.count()
)

print(
    "RF features:",
    rf_xai.count()
)

print(
    "MLP SHAP features:",
    mlp_xai.count()
)


# 5. Standardize model importance tables
#
# We cannot directly compare raw coefficient values,
# tree importance values and SHAP values because they are
# measured on different scales.
#
# Therefore:
#
# LR:
# absolute coefficient
#
# DT:
# feature importance
#
# RF:
# feature importance
#
# MLP:
# mean absolute SHAP
#
# Later each model's importance is normalized so that
# its total importance equals 1.

lr_importance = (

    lr_xai

    .select(
        col(
            "feature"
        ),

        col(
            "absolute_coefficient"
        ).alias(
            "raw_importance"
        )
    )

    .withColumn(
        "model",
        lit(
            "Logistic Regression"
        )
    )
)


dt_importance = (

    dt_xai

    .select(
        col(
            "feature"
        ),

        col(
            "importance"
        ).alias(
            "raw_importance"
        )
    )

    .withColumn(
        "model",
        lit(
            "Decision Tree"
        )
    )
)


rf_importance = (

    rf_xai

    .select(
        col(
            "feature"
        ),

        col(
            "importance"
        ).alias(
            "raw_importance"
        )
    )

    .withColumn(
        "model",
        lit(
            "Random Forest"
        )
    )
)


mlp_importance = (

    mlp_xai

    .select(
        col(
            "transformed_feature"
        ).alias(
            "feature"
        ),

        col(
            "mean_absolute_shap"
        ).alias(
            "raw_importance"
        )
    )

    .withColumn(
        "model",
        lit(
            "MLP SHAP"
        )
    )
)


# 6. Combine all model explanation results

xai_long = (

    lr_importance

    .unionByName(
        dt_importance
    )

    .unionByName(
        rf_importance
    )

    .unionByName(
        mlp_importance
    )
)


print(
    "Combined explanation table created."
)


# 7. Normalize importance within each model
#
# normalized importance =
#
# individual feature importance
# total importance for that model
#
# This allows approximate cross-model magnitude comparison.

model_window = (
    Window
    .partitionBy(
        "model"
    )
)


xai_long = (

    xai_long

    .withColumn(
        "model_total_importance",

        spark_sum(
            "raw_importance"
        ).over(
            model_window
        )
    )

    .withColumn(
        "normalized_importance",

        when(
            col(
                "model_total_importance"
            ) > 0,

            col(
                "raw_importance"
            )
            /
            col(
                "model_total_importance"
            )
        )

        .otherwise(
            lit(0.0)
        )
    )
)


# 8. Rank features independently within each model
#
# feature name is used as secondary ordering so that ties
# remain deterministic and exactly 10 / 20 features can
# be selected later.

rank_window = (

    Window

    .partitionBy(
        "model"
    )

    .orderBy(
        desc(
            "raw_importance"
        ),
        asc(
            "feature"
        )
    )
)


xai_long = (

    xai_long

    .withColumn(
        "rank",

        row_number().over(
            rank_window
        )
    )
)


print(
    "Feature rankings calculated."
)


display(
    xai_long
    .orderBy(
        "model",
        "rank"
    )
    .filter(
        col(
            "rank"
        )
        <= 10
    )
)


# 9. Save complete long-format XAI comparison

xai_long.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_fico_all_model_importance"
    )


print(
    "Saved: "
    "lc_xai_fico_all_model_importance"
)


# 10. Create model-specific ranking tables

lr_ranked = (

    xai_long

    .filter(
        col(
            "model"
        )
        ==
        "Logistic Regression"
    )

    .select(
        "feature",

        col(
            "raw_importance"
        ).alias(
            "lr_importance"
        ),

        col(
            "normalized_importance"
        ).alias(
            "lr_normalized_importance"
        ),

        col(
            "rank"
        ).alias(
            "lr_rank"
        )
    )
)


dt_ranked = (

    xai_long

    .filter(
        col(
            "model"
        )
        ==
        "Decision Tree"
    )

    .select(
        "feature",

        col(
            "raw_importance"
        ).alias(
            "dt_importance"
        ),

        col(
            "normalized_importance"
        ).alias(
            "dt_normalized_importance"
        ),

        col(
            "rank"
        ).alias(
            "dt_rank"
        )
    )
)


rf_ranked = (

    xai_long

    .filter(
        col(
            "model"
        )
        ==
        "Random Forest"
    )

    .select(
        "feature",

        col(
            "raw_importance"
        ).alias(
            "rf_importance"
        ),

        col(
            "normalized_importance"
        ).alias(
            "rf_normalized_importance"
        ),

        col(
            "rank"
        ).alias(
            "rf_rank"
        )
    )
)


mlp_ranked = (

    xai_long

    .filter(
        col(
            "model"
        )
        ==
        "MLP SHAP"
    )

    .select(
        "feature",

        col(
            "raw_importance"
        ).alias(
            "mlp_shap_importance"
        ),

        col(
            "normalized_importance"
        ).alias(
            "mlp_normalized_importance"
        ),

        col(
            "rank"
        ).alias(
            "mlp_rank"
        )
    )
)


# 11. Join all models into one feature-comparison table

feature_comparison = (

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

    .join(
        mlp_ranked,
        on="feature",
        how="outer"
    )
)


# 12. Calculate top-20 agreement

feature_comparison = (

    feature_comparison

    .withColumn(
        "lr_top20",

        when(
            col(
                "lr_rank"
            )
            <= 20,
            1
        )
        .otherwise(
            0
        )
    )

    .withColumn(
        "dt_top20",

        when(
            col(
                "dt_rank"
            )
            <= 20,
            1
        )
        .otherwise(
            0
        )
    )

    .withColumn(
        "rf_top20",

        when(
            col(
                "rf_rank"
            )
            <= 20,
            1
        )
        .otherwise(
            0
        )
    )

    .withColumn(
        "mlp_top20",

        when(
            col(
                "mlp_rank"
            )
            <= 20,
            1
        )
        .otherwise(
            0
        )
    )

    .withColumn(
        "models_in_top20",

        col(
            "lr_top20"
        )
        +
        col(
            "dt_top20"
        )
        +
        col(
            "rf_top20"
        )
        +
        col(
            "mlp_top20"
        )
    )
)


# 13. Calculate average feature rank
#
# Because the models have the same 135-dimensional feature
# space, average rank provides a simple consensus indicator.

feature_comparison = (

    feature_comparison

    .withColumn(
        "average_rank",

        (
            col(
                "lr_rank"
            )
            +
            col(
                "dt_rank"
            )
            +
            col(
                "rf_rank"
            )
            +
            col(
                "mlp_rank"
            )
        )
        / 4.0
    )
)


# 14. Average normalized importance

feature_comparison = (

    feature_comparison

    .withColumn(
        "average_normalized_importance",

        (
            col(
                "lr_normalized_importance"
            )
            +
            col(
                "dt_normalized_importance"
            )
            +
            col(
                "rf_normalized_importance"
            )
            +
            col(
                "mlp_normalized_importance"
            )
        )
        / 4.0
    )
)


# 15. Add simple consensus category

feature_comparison = (

    feature_comparison

    .withColumn(
        "consensus_level",

        when(
            col(
                "models_in_top20"
            )
            == 4,
            "Very High"
        )

        .when(
            col(
                "models_in_top20"
            )
            == 3,
            "High"
        )

        .when(
            col(
                "models_in_top20"
            )
            == 2,
            "Moderate"
        )

        .when(
            col(
                "models_in_top20"
            )
            == 1,
            "Low"
        )

        .otherwise(
            "None"
        )
    )
)


# 16. Display strongest transformed-feature consensus

print(
    "Strongest cross-model "
    "feature agreement:"
)


display(

    feature_comparison

    .orderBy(
        desc(
            "models_in_top20"
        ),
        asc(
            "average_rank"
        )
    )

    .limit(
        TOP_FEATURES_TO_DISPLAY
    )
)


# 17. Save complete feature comparison

feature_comparison.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_fico_feature_comparison"
    )


print(
    "Saved: "
    "lc_xai_fico_feature_comparison"
)


# 18. Save consensus features
#
# Keep features that appear in the top 20 of at least
# two of the four explanation methods.

consensus_features = (

    feature_comparison

    .filter(
        col(
            "models_in_top20"
        )
        >= 2
    )

    .orderBy(
        desc(
            "models_in_top20"
        ),
        asc(
            "average_rank"
        )
    )
)


print(
    "Features appearing in the "
    "top 20 of at least two models:"
)


display(
    consensus_features
)


consensus_features.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_fico_consensus_features"
    )


# 19. Map transformed feature names back to original variables
#
# This is used only for a business-level consensus table.
#
# We use the BEST rank of any encoded component rather than
# summing category importances, avoiding automatic inflation
# of variables such as addr_state simply because they contain
# many one-hot dimensions.

def original_feature_expression(
    feature_column
):

    return (

        when(
            feature_column.startswith(
                "home_ownership_encoded_"
            ),
            lit(
                "home_ownership"
            )
        )

        .when(
            feature_column.startswith(
                "verification_status_encoded_"
            ),
            lit(
                "verification_status"
            )
        )

        .when(
            feature_column.startswith(
                "purpose_encoded_"
            ),
            lit(
                "purpose"
            )
        )

        .when(
            feature_column.startswith(
                "addr_state_encoded_"
            ),
            lit(
                "addr_state"
            )
        )

        .when(
            feature_column.startswith(
                "initial_list_status_encoded_"
            ),
            lit(
                "initial_list_status"
            )
        )

        .when(
            feature_column.startswith(
                "application_type_encoded_"
            ),
            lit(
                "application_type"
            )
        )

        .otherwise(
            feature_column
        )
    )


feature_comparison_business = (

    feature_comparison

    .withColumn(
        "original_feature",

        original_feature_expression(
            col(
                "feature"
            )
        )
    )
)


# 20. Original-feature consensus using best component rank

original_feature_consensus = (

    feature_comparison_business

    .groupBy(
        "original_feature"
    )

    .agg(

        spark_min(
            "lr_rank"
        ).alias(
            "lr_best_rank"
        ),

        spark_min(
            "dt_rank"
        ).alias(
            "dt_best_rank"
        ),

        spark_min(
            "rf_rank"
        ).alias(
            "rf_best_rank"
        ),

        spark_min(
            "mlp_rank"
        ).alias(
            "mlp_best_rank"
        )
    )
)


original_feature_consensus = (

    original_feature_consensus

    .withColumn(
        "models_with_top20_component",

        when(
            col(
                "lr_best_rank"
            )
            <= 20,
            1
        )
        .otherwise(
            0
        )

        +

        when(
            col(
                "dt_best_rank"
            )
            <= 20,
            1
        )
        .otherwise(
            0
        )

        +

        when(
            col(
                "rf_best_rank"
            )
            <= 20,
            1
        )
        .otherwise(
            0
        )

        +

        when(
            col(
                "mlp_best_rank"
            )
            <= 20,
            1
        )
        .otherwise(
            0
        )
    )
)


original_feature_consensus = (

    original_feature_consensus

    .withColumn(
        "average_best_rank",

        (
            col(
                "lr_best_rank"
            )
            +
            col(
                "dt_best_rank"
            )
            +
            col(
                "rf_best_rank"
            )
            +
            col(
                "mlp_best_rank"
            )
        )
        / 4.0
    )

    .orderBy(
        desc(
            "models_with_top20_component"
        ),
        asc(
            "average_best_rank"
        )
    )
)


print(
    "Original business-variable "
    "cross-model consensus:"
)


display(
    original_feature_consensus
    .limit(
        TOP_FEATURES_TO_DISPLAY
    )
)


original_feature_consensus.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_fico_original_feature_consensus"
    )


# 21. Collect feature rankings for pairwise agreement
#
# Only 135 features are involved, so collecting this small
# table to Python is safe.

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "MLP SHAP"
]


model_feature_rankings = {}


for model_name in MODEL_NAMES:

    ranked_features = (

        xai_long

        .filter(
            col(
                "model"
            )
            ==
            model_name
        )

        .orderBy(
            "rank"
        )

        .select(
            "feature"
        )

        .collect()
    )


    model_feature_rankings[
        model_name
    ] = [

        row[
            "feature"
        ]

        for row
        in ranked_features
    ]


# 22. Calculate pairwise top-k Jaccard agreement
#
# Jaccard =
#
# number of shared features
# -------------------------
# number of features in union
#
# 0 = no overlap
# 1 = identical feature sets

jaccard_rows = []


for top_k in TOP_K_VALUES:

    for (
        model_a,
        model_b
    ) in itertools.combinations(
        MODEL_NAMES,
        2
    ):

        set_a = set(
            model_feature_rankings[
                model_a
            ][:top_k]
        )

        set_b = set(
            model_feature_rankings[
                model_b
            ][:top_k]
        )


        intersection = (
            set_a
            .intersection(
                set_b
            )
        )


        union = (
            set_a
            .union(
                set_b
            )
        )


        jaccard_score = (

            len(
                intersection
            )
            /
            len(
                union
            )

            if len(
                union
            ) > 0

            else 0.0
        )


        shared_features = (
            ", ".join(
                sorted(
                    intersection
                )
            )
        )


        jaccard_rows.append(
            (
                model_a,
                model_b,
                int(
                    top_k
                ),
                int(
                    len(
                        intersection
                    )
                ),
                int(
                    len(
                        union
                    )
                ),
                float(
                    jaccard_score
                ),
                shared_features
            )
        )


jaccard_df = (
    spark.createDataFrame(
        jaccard_rows,
        [
            "model_a",
            "model_b",
            "top_k",
            "shared_feature_count",
            "union_feature_count",
            "jaccard_similarity",
            "shared_features"
        ]
    )
)


print(
    "Pairwise top-k explanation agreement:"
)


display(
    jaccard_df
    .orderBy(
        "top_k",
        desc(
            "jaccard_similarity"
        )
    )
)


jaccard_df.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_fico_pairwise_jaccard"
    )


# 23. Calculate pairwise feature-rank correlations
#
# Spearman correlation is equivalent to Pearson correlation
# performed on ranked observations.
#
# Here ties are handled using average ranks.

importance_pd = (

    xai_long

    .select(
        "feature",
        "model",
        "raw_importance"
    )

    .toPandas()
)


importance_wide_pd = (

    importance_pd

    .pivot(
        index="feature",
        columns="model",
        values="raw_importance"
    )

    .fillna(
        0.0
    )
)


rank_pd = pd.DataFrame(
    index=(
        importance_wide_pd
        .index
    )
)


for model_name in MODEL_NAMES:

    rank_pd[
        model_name
    ] = (

        importance_wide_pd[
            model_name
        ]

        .rank(
            method="average",
            ascending=False
        )
    )


rank_correlation_rows = []


for (
    model_a,
    model_b
) in itertools.combinations(
    MODEL_NAMES,
    2
):

    correlation = float(
        np.corrcoef(
            rank_pd[
                model_a
            ],
            rank_pd[
                model_b
            ]
        )[0, 1]
    )


    rank_correlation_rows.append(
        (
            model_a,
            model_b,
            correlation
        )
    )


rank_correlation_df = (
    spark.createDataFrame(
        rank_correlation_rows,
        [
            "model_a",
            "model_b",
            "spearman_rank_correlation"
        ]
    )
)


print(
    "Pairwise explanation "
    "rank correlations:"
)


display(
    rank_correlation_df
    .orderBy(
        desc(
            "spearman_rank_correlation"
        )
    )
)


rank_correlation_df.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_fico_rank_correlation"
    )


# 24. Plot pairwise Top-20 Jaccard similarity

top20_jaccard_pd = (

    jaccard_df

    .filter(
        col(
            "top_k"
        )
        == 20
    )

    .select(
        "model_a",
        "model_b",
        "jaccard_similarity"
    )

    .toPandas()
)


jaccard_matrix = pd.DataFrame(

    np.eye(
        len(
            MODEL_NAMES
        )
    ),

    index=MODEL_NAMES,
    columns=MODEL_NAMES
)


for _, row in (
    top20_jaccard_pd
    .iterrows()
):

    jaccard_matrix.loc[
        row[
            "model_a"
        ],
        row[
            "model_b"
        ]
    ] = row[
        "jaccard_similarity"
    ]


    jaccard_matrix.loc[
        row[
            "model_b"
        ],
        row[
            "model_a"
        ]
    ] = row[
        "jaccard_similarity"
    ]


plt.figure(
    figsize=(
        8,
        7
    )
)


plt.imshow(
    jaccard_matrix.values
)


plt.xticks(
    range(
        len(
            MODEL_NAMES
        )
    ),
    MODEL_NAMES,
    rotation=45,
    ha="right"
)


plt.yticks(
    range(
        len(
            MODEL_NAMES
        )
    ),
    MODEL_NAMES
)


plt.colorbar(
    label=(
        "Top-20 Jaccard similarity"
    )
)


for row_index in range(
    len(
        MODEL_NAMES
    )
):

    for column_index in range(
        len(
            MODEL_NAMES
        )
    ):

        plt.text(
            column_index,
            row_index,
            (
                f"{jaccard_matrix.iloc[row_index, column_index]:.2f}"
            ),
            ha="center",
            va="center"
        )


plt.title(
    "Top-20 Feature Agreement Between XAI Methods"
)


plt.tight_layout()


jaccard_plot_path = (
    os.path.join(
        OUTPUT_DIR,
        "xai_top20_jaccard_similarity.png"
    )
)


plt.savefig(
    jaccard_plot_path,
    dpi=200,
    bbox_inches="tight"
)


plt.show()
plt.close()


print(
    "Saved:",
    jaccard_plot_path
)


# 25. Plot rank-correlation matrix

rank_correlation_matrix = pd.DataFrame(

    np.eye(
        len(
            MODEL_NAMES
        )
    ),

    index=MODEL_NAMES,
    columns=MODEL_NAMES
)


for row in (
    rank_correlation_rows
):

    model_a = row[0]
    model_b = row[1]
    correlation = row[2]


    rank_correlation_matrix.loc[
        model_a,
        model_b
    ] = correlation


    rank_correlation_matrix.loc[
        model_b,
        model_a
    ] = correlation


plt.figure(
    figsize=(
        8,
        7
    )
)


plt.imshow(
    rank_correlation_matrix.values,
    vmin=-1,
    vmax=1
)


plt.xticks(
    range(
        len(
            MODEL_NAMES
        )
    ),
    MODEL_NAMES,
    rotation=45,
    ha="right"
)


plt.yticks(
    range(
        len(
            MODEL_NAMES
        )
    ),
    MODEL_NAMES
)


plt.colorbar(
    label=(
        "Feature-rank correlation"
    )
)


for row_index in range(
    len(
        MODEL_NAMES
    )
):

    for column_index in range(
        len(
            MODEL_NAMES
        )
    ):

        plt.text(
            column_index,
            row_index,
            (
                f"{rank_correlation_matrix.iloc[row_index, column_index]:.2f}"
            ),
            ha="center",
            va="center"
        )


plt.title(
    "Global XAI Feature-Ranking Agreement"
)


plt.tight_layout()


correlation_plot_path = (
    os.path.join(
        OUTPUT_DIR,
        "xai_feature_rank_correlation.png"
    )
)


plt.savefig(
    correlation_plot_path,
    dpi=200,
    bbox_inches="tight"
)


plt.show()
plt.close()


print(
    "Saved:",
    correlation_plot_path
)


# 26. Plot strongest original-feature consensus

original_consensus_pd = (

    original_feature_consensus

    .limit(
        TOP_FEATURES_TO_DISPLAY
    )

    .toPandas()
)


original_consensus_pd = (
    original_consensus_pd
    .iloc[::-1]
)


plt.figure(
    figsize=(
        9,
        8
    )
)


plt.barh(
    original_consensus_pd[
        "original_feature"
    ],
    original_consensus_pd[
        "models_with_top20_component"
    ]
)


plt.xlabel(
    "Number of models with feature in Top 20"
)


plt.ylabel(
    "Original feature"
)


plt.title(
    "Cross-Model Global XAI Consensus"
)


plt.xlim(
    0,
    4.2
)


plt.tight_layout()


consensus_plot_path = (
    os.path.join(
        OUTPUT_DIR,
        "xai_cross_model_consensus.png"
    )
)


plt.savefig(
    consensus_plot_path,
    dpi=200,
    bbox_inches="tight"
)


plt.show()
plt.close()


print(
    "Saved:",
    consensus_plot_path
)


# 27. Load final FICO model performance tables

lr_metrics = spark.table(
    "dissertation.lendingclub."
    "lc_2007_2017_logistic_regression_fico_metrics"
)


dt_metrics = spark.table(
    "dissertation.lendingclub."
    "lc_2007_2017_decision_tree_fico_metrics"
)


rf_metrics = spark.table(
    "dissertation.lendingclub."
    "lc_2007_2017_random_forest_fico_metrics"
)


mlp_metrics = spark.table(
    "dissertation.lendingclub."
    "lc_2007_2017_deep_learning_mlp_fico_metrics"
)


# 28. Canonical model names for performance comparison

lr_performance = (

    lr_metrics

    .select(
        "auc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "default_precision",
        "default_recall",
        "default_f1"
    )

    .withColumn(
        "model",
        lit(
            "Logistic Regression"
        )
    )
)


dt_performance = (

    dt_metrics

    .select(
        "auc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "default_precision",
        "default_recall",
        "default_f1"
    )

    .withColumn(
        "model",
        lit(
            "Decision Tree"
        )
    )
)


rf_performance = (

    rf_metrics

    .select(
        "auc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "default_precision",
        "default_recall",
        "default_f1"
    )

    .withColumn(
        "model",
        lit(
            "Random Forest"
        )
    )
)


mlp_performance = (

    mlp_metrics

    .select(
        "auc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "default_precision",
        "default_recall",
        "default_f1"
    )

    .withColumn(
        "model",
        lit(
            "Multilayer Perceptron"
        )
    )
)


performance_df = (

    lr_performance

    .unionByName(
        dt_performance
    )

    .unionByName(
        rf_performance
    )

    .unionByName(
        mlp_performance
    )
)


# 29. Rank predictive performance

auc_window = (
    Window.orderBy(
        desc(
            "auc"
        )
    )
)


default_recall_window = (
    Window.orderBy(
        desc(
            "default_recall"
        )
    )
)


default_f1_window = (
    Window.orderBy(
        desc(
            "default_f1"
        )
    )
)


performance_df = (

    performance_df

    .withColumn(
        "auc_rank",

        row_number().over(
            auc_window
        )
    )

    .withColumn(
        "default_recall_rank",

        row_number().over(
            default_recall_window
        )
    )

    .withColumn(
        "default_f1_rank",

        row_number().over(
            default_f1_window
        )
    )
)


# 30. Create interpretability methodology table
#
# These are qualitative categories, not numerical performance
# measurements.

interpretability_rows = [

    (
        "Logistic Regression",
        "Intrinsic",
        "Coefficients",
        "Global and directional",
        "High"
    ),

    (
        "Decision Tree",
        "Intrinsic",
        "Feature importance and decision rules",
        "Global and rule-level",
        "High"
    ),

    (
        "Random Forest",
        "Intrinsic / model-specific",
        "Aggregated feature importance",
        "Primarily global",
        "Medium"
    ),

    (
        "Multilayer Perceptron",
        "Post-hoc",
        "Kernel SHAP",
        "Global sampled and local",
        "Low intrinsic / Medium post-hoc"
    )
]


interpretability_df = (
    spark.createDataFrame(
        interpretability_rows,
        [
            "model",
            "explanation_type",
            "explanation_method",
            "explanation_scope",
            "interpretability_level"
        ]
    )
)


# 31. Combine predictive performance and interpretability

performance_interpretability = (

    performance_df

    .join(
        interpretability_df,
        on="model",
        how="left"
    )

    .orderBy(
        desc(
            "auc"
        )
    )
)


print(
    "Final performance vs interpretability comparison:"
)


display(
    performance_interpretability
)


performance_interpretability.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_fico_performance_interpretability"
    )


# 32. Save XAI evaluation methodology/configuration

evaluation_config_rows = [

    (
        4,
        10,
        20,
        135,
        (
            "Absolute LR coefficients, "
            "DT feature importance, "
            "RF feature importance, "
            "MLP mean absolute Kernel SHAP"
        ),
        (
            "Pairwise Top-k Jaccard overlap "
            "and feature-rank correlation"
        )
    )
]


evaluation_config_df = (
    spark.createDataFrame(
        evaluation_config_rows,
        [
            "number_of_models",
            "top_k_1",
            "top_k_2",
            "transformed_feature_count",
            "importance_sources",
            "agreement_methods"
        ]
    )
)


evaluation_config_df.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_fico_evaluation_config"
    )


# 33. Automatically identify key final findings

best_auc_model = (

    performance_interpretability

    .orderBy(
        desc(
            "auc"
        )
    )

    .first()
)


best_default_recall_model = (

    performance_interpretability

    .orderBy(
        desc(
            "default_recall"
        )
    )

    .first()
)


best_default_f1_model = (

    performance_interpretability

    .orderBy(
        desc(
            "default_f1"
        )
    )

    .first()
)


strongest_consensus_features = (

    original_feature_consensus

    .orderBy(
        desc(
            "models_with_top20_component"
        ),
        asc(
            "average_best_rank"
        )
    )

    .limit(
        10
    )

    .collect()
)


print(
    "\n"
    "=================================================="
)

print(
    "FINAL XAI EVALUATION SUMMARY"
)

print(
    "=================================================="
)


print(
    "\nBest AUC:"
)

print(
    best_auc_model[
        "model"
    ],
    "=",
    round(
        best_auc_model[
            "auc"
        ],
        4
    )
)


print(
    "\nBest default recall:"
)

print(
    best_default_recall_model[
        "model"
    ],
    "=",
    round(
        best_default_recall_model[
            "default_recall"
        ],
        4
    )
)


print(
    "\nBest default F1:"
)

print(
    best_default_f1_model[
        "model"
    ],
    "=",
    round(
        best_default_f1_model[
            "default_f1"
        ],
        4
    )
)


print(
    "\nStrongest cross-model "
    "original-feature consensus:"
)


for row in (
    strongest_consensus_features
):

    print(
        row[
            "original_feature"
        ],
        "| models in top 20:",
        row[
            "models_with_top20_component"
        ],
        "| average best rank:",
        round(
            row[
                "average_best_rank"
            ],
            2
        )
    )


# 34. Final output summary

print(
    "\n"
    "10_xai_evaluation.py "
    "completed successfully."
)


print(
    "\nCreated Delta tables:"
)


print(
    "1. dissertation.lendingclub."
    "lc_xai_fico_all_model_importance"
)


print(
    "2. dissertation.lendingclub."
    "lc_xai_fico_feature_comparison"
)


print(
    "3. dissertation.lendingclub."
    "lc_xai_fico_consensus_features"
)


print(
    "4. dissertation.lendingclub."
    "lc_xai_fico_original_feature_consensus"
)


print(
    "5. dissertation.lendingclub."
    "lc_xai_fico_pairwise_jaccard"
)


print(
    "6. dissertation.lendingclub."
    "lc_xai_fico_rank_correlation"
)


print(
    "7. dissertation.lendingclub."
    "lc_xai_fico_performance_interpretability"
)


print(
    "8. dissertation.lendingclub."
    "lc_xai_fico_evaluation_config"
)


print(
    "\nFigures saved in:"
)


print(
    OUTPUT_DIR
)


print(
    "\nExpected figure files:"
)


print(
    "xai_top20_jaccard_similarity.png"
)

print(
    "xai_feature_rank_correlation.png"
)

print(
    "xai_cross_model_consensus.png"
)