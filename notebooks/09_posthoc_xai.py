# Main objectives:
# 1. Load the final saved MLP FICO model
# 2. Generate sampled global Kernel SHAP explanations
# 3. Aggregate one-hot features back to original business features
# 4. Explain high-confidence TP / TN / FP / FN cases
# 5. Retain original borrower values for interpretation
# 6. Save final XAI tables and figures

# LIME is intentionally excluded from the final analysis because
# previous experiments produced unstable / effectively zero local
# surrogate coefficients for several diagnostic cases.




# 1. Imports


!pip install shap
import os

import numpy as np
import matplotlib.pyplot as plt
import shap

from pyspark.ml import PipelineModel

from pyspark.ml.classification import (
    MultilayerPerceptronClassificationModel
)

from pyspark.ml.linalg import Vectors

from pyspark.ml.functions import (
    vector_to_array
)

from pyspark.sql.functions import (
    col,
    desc,
    asc
)



# 2. Configuration


DATA_TABLE = (
    "dissertation.lendingclub."
    "lc_2007_2017_ml_no_leakage_fico"
)


PREPROCESSING_MODEL_PATH = (
    "dbfs:/Volumes/dissertation/lendingclub/"
    "model_artifacts/"
    "lc_2007_2017_deep_learning_mlp_fico_preprocessing"
)


MLP_MODEL_PATH = (
    "dbfs:/Volumes/dissertation/lendingclub/"
    "model_artifacts/"
    "lc_2007_2017_deep_learning_mlp_fico_model"
)


RANDOM_SEED = 42

# SHAP configuration----

# Pool of actual training observations used before k-means
BACKGROUND_POOL_ROWS = 500

# Representative background profiles used by Kernel SHAP
BACKGROUND_CLUSTERS = 10

# Held-out observations used for sampled global SHAP
# GLOBAL_EXPLAIN_ROWS = 30
GLOBAL_EXPLAIN_ROWS = 100

# Number of Kernel SHAP perturbation samples
# SHAP_NSAMPLES = 100
SHAP_NSAMPLES = 200
# Features displayed in global/local outputs
TOP_FEATURES = 20

# Maximum number of records passed through Spark in one prediction batch
PREDICTION_BATCH_SIZE = 2000



# 3. Output location


spark.sql(
    """
    CREATE VOLUME IF NOT EXISTS
    dissertation.lendingclub.xai_outputs
    """
)


OUTPUT_DIR = (
    "/Volumes/dissertation/"
    "lendingclub/xai_outputs"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print(
    "XAI output directory:",
    OUTPUT_DIR
)



# 4. Load final FICO dataset


df = spark.table(
    DATA_TABLE
)


print(
    "Number of rows:",
    df.count()
)

print(
    "Number of columns:",
    len(df.columns)
)



# 5. Load saved preprocessing pipeline and MLP model


preprocessing_model = (
    PipelineModel
    .load(
        PREPROCESSING_MODEL_PATH
    )
)


mlp_model = (
    MultilayerPerceptronClassificationModel
    .load(
        MLP_MODEL_PATH
    )
)


print(
    "Saved MLP preprocessing model loaded."
)

print(
    "Saved MLP FICO model loaded."
)

print(
    "MLP architecture:",
    mlp_model.getLayers()
)

print(
    "Number of MLP features:",
    mlp_model.numFeatures
)



# 6. Recreate final train/test split


train_df, test_df = (
    df.randomSplit(
        [0.8, 0.2],
        seed=RANDOM_SEED
    )
)


print(
    "Train rows:",
    train_df.count()
)

print(
    "Test rows:",
    test_df.count()
)



# 7. Apply saved preprocessing


train_prepared = (
    preprocessing_model
    .transform(
        train_df
    )
)


test_prepared = (
    preprocessing_model
    .transform(
        test_df
    )
)


print(
    "Saved preprocessing pipeline applied."
)


# 8. Recover model feature names

def get_feature_names(
    prepared_df
):

    possible_vector_cols = [
        "features_unscaled",
        "features"
    ]

    for vector_col in possible_vector_cols:

        if vector_col not in prepared_df.columns:
            continue

        metadata = (
            prepared_df
            .schema[vector_col]
            .metadata
        )

        attrs = (
            metadata
            .get(
                "ml_attr",
                {}
            )
            .get(
                "attrs",
                {}
            )
        )

        feature_info = []

        for attr_type in [
            "numeric",
            "binary",
            "nominal"
        ]:

            feature_info.extend(
                attrs.get(
                    attr_type,
                    []
                )
            )

        if len(feature_info) > 0:

            feature_info = sorted(
                feature_info,
                key=lambda x: x["idx"]
            )

            return [
                item["name"]
                for item
                in feature_info
            ]

    # Fallback

    first_vector = (
        prepared_df
        .select("features")
        .first()["features"]
    )

    feature_count = (
        first_vector.size
    )

    return [
        f"feature_{i}"
        for i
        in range(
            feature_count
        )
    ]


sample_prepared = (
    preprocessing_model
    .transform(
        df.limit(1)
    )
)


feature_names = (
    get_feature_names(
        sample_prepared
    )
)


print(
    "Recovered feature names:",
    len(feature_names)
)


if (
    len(feature_names)
    != mlp_model.numFeatures
):

    raise ValueError(
        "Feature-name count does not "
        "match MLP input dimension."
    )


print(
    "Feature-name mapping verified."
)



# 9. Map one-hot features back to original variables


CATEGORY_PREFIX_MAP = {

    "home_ownership_encoded_":
        "home_ownership",

    "verification_status_encoded_":
        "verification_status",

    "purpose_encoded_":
        "purpose",

    "addr_state_encoded_":
        "addr_state",

    "initial_list_status_encoded_":
        "initial_list_status",

    "application_type_encoded_":
        "application_type"
}


def get_original_feature_group(
    transformed_feature
):

    for (
        prefix,
        original_feature
    ) in CATEGORY_PREFIX_MAP.items():

        if transformed_feature.startswith(
            prefix
        ):

            return original_feature

    return transformed_feature


def get_encoded_category_info(
    transformed_feature
):

    """
    Returns:
        original feature,
        encoded category

    Example:
        purpose_encoded_small_business

    becomes:
        purpose,
        small_business
    """

    for (
        prefix,
        original_feature
    ) in CATEGORY_PREFIX_MAP.items():

        if transformed_feature.startswith(
            prefix
        ):

            encoded_category = (
                transformed_feature[
                    len(prefix):
                ]
            )

            return (
                original_feature,
                encoded_category
            )

    return (
        transformed_feature,
        None
    )



# 10. Score test dataset with final MLP model


test_scored = (
    mlp_model
    .transform(
        test_prepared
    )
    .withColumn(
        "default_probability",
        vector_to_array(
            col("probability")
        )[1]
    )
)


print(
    "Test data scored."
)



# 11. Select high-confidence diagnostic cases
#
# TP = actual default, predicted default
# TN = actual non-default, predicted non-default
# FP = actual non-default, predicted default
# FN = actual default, predicted non-default
#
# We deliberately select high-confidence examples because they
# are useful for investigating why the model is confidently
# correct or confidently wrong.


def select_case(
    case_type,
    actual_value,
    prediction_value,
    probability_order
):

    case_df = (
        test_scored
        .filter(
            (
                col("default_flag")
                == actual_value
            )
            &
            (
                col("prediction")
                == prediction_value
            )
        )
    )

    if probability_order == "desc":

        case_df = (
            case_df
            .orderBy(
                desc(
                    "default_probability"
                )
            )
        )

    else:

        case_df = (
            case_df
            .orderBy(
                asc(
                    "default_probability"
                )
            )
        )


    row = case_df.first()


    if row is None:

        print(
            "No observation found for:",
            case_type
        )

        return None



    # Save the original ML-table values BEFORE interpretation


    raw_values = {

        column_name:
            row[column_name]

        for column_name
        in df.columns
    }


    return {

        "case_type":
            case_type,

        "actual":
            int(
                row["default_flag"]
            ),

        "prediction":
            int(
                row["prediction"]
            ),

        "default_probability":
            float(
                row[
                    "default_probability"
                ]
            ),

        "features":
            row["features"],

        "raw_values":
            raw_values
    }


case_specs = [

    (
        "true_positive",
        1,
        1,
        "desc"
    ),

    (
        "true_negative",
        0,
        0,
        "asc"
    ),

    (
        "false_positive",
        0,
        1,
        "desc"
    ),

    (
        "false_negative",
        1,
        0,
        "asc"
    )
]


cases = []


for case_spec in case_specs:

    selected_case = (
        select_case(
            *case_spec
        )
    )

    if selected_case is not None:

        cases.append(
            selected_case
        )



# 12. Save diagnostic case summary


case_summary_rows = [

    (
        case["case_type"],
        case["actual"],
        case["prediction"],
        case[
            "default_probability"
        ]
    )

    for case in cases
]


case_summary_df = (
    spark.createDataFrame(
        case_summary_rows,
        [
            "case_type",
            "actual_default_flag",
            "prediction",
            "default_probability"
        ]
    )
)


print(
    "High-confidence diagnostic cases:"
)


display(
    case_summary_df
)


case_summary_df.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_mlp_fico_local_cases"
    )


print(
    "Saved diagnostic case table."
)



# 13. Convert Spark feature vectors to NumPy


def feature_vectors_to_numpy(
    spark_df,
    number_of_rows
):

    rows = (
        spark_df
        .select("features")
        .limit(
            number_of_rows
        )
        .collect()
    )


    if len(rows) == 0:

        raise ValueError(
            "No feature vectors collected."
        )


    return np.vstack(
        [
            row[
                "features"
            ].toArray()

            for row
            in rows
        ]
    )



# 14. Create SHAP background pool


background_pool_df = (

    train_prepared

    .sample(
        withReplacement=False,
        fraction=0.01,
        seed=RANDOM_SEED
    )

    .limit(
        BACKGROUND_POOL_ROWS
    )
)


background_pool_np = (
    feature_vectors_to_numpy(
        background_pool_df,
        BACKGROUND_POOL_ROWS
    )
)


print(
    "Background pool:",
    background_pool_np.shape
)



# 15. Summarize background with k-means


background_summary = (
    shap.kmeans(
        background_pool_np,
        BACKGROUND_CLUSTERS
    )
)


print(
    "SHAP k-means clusters:",
    BACKGROUND_CLUSTERS
)



# 16. Create held-out global explanation sample


global_explain_df = (

    test_scored

    .sample(
        withReplacement=False,
        fraction=0.01,
        seed=RANDOM_SEED
    )

    .limit(
        GLOBAL_EXPLAIN_ROWS
    )
)


global_explain_np = (
    feature_vectors_to_numpy(
        global_explain_df,
        GLOBAL_EXPLAIN_ROWS
    )
)


print(
    "Global SHAP sample shape:",
    global_explain_np.shape
)



# 17. Batched Spark MLP probability prediction


def mlp_predict_proba_numpy(
    X
):

    X = np.asarray(
        X,
        dtype=float
    )


    if X.ndim == 1:

        X = X.reshape(
            1,
            -1
        )


    all_predictions = []


    for batch_start in range(
        0,
        len(X),
        PREDICTION_BATCH_SIZE
    ):

        batch_end = min(
            batch_start
            +
            PREDICTION_BATCH_SIZE,

            len(X)
        )


        batch = (
            X[
                batch_start:
                batch_end
            ]
        )


        spark_rows = [

            (
                int(
                    batch_start
                    +
                    row_index
                ),

                Vectors.dense(
                    row.tolist()
                )
            )

            for (
                row_index,
                row
            )
            in enumerate(
                batch
            )
        ]


        batch_df = (
            spark.createDataFrame(
                spark_rows,
                [
                    "row_id",
                    "features"
                ]
            )
        )


        batch_predictions = (

            mlp_model

            .transform(
                batch_df
            )

            .select(
                "row_id",

                vector_to_array(
                    col(
                        "probability"
                    )
                )
                .alias(
                    "probability_array"
                )
            )

            .collect()
        )


        for prediction_row in (
            batch_predictions
        ):

            all_predictions.append(
                (
                    int(
                        prediction_row[
                            "row_id"
                        ]
                    ),

                    prediction_row[
                        "probability_array"
                    ]
                )
            )


    # Restore original row order
    all_predictions = sorted(
        all_predictions,
        key=lambda x: x[0]
    )


    probability_array = np.asarray(
        [
            probability

            for (
                row_id,
                probability
            )
            in all_predictions
        ]
    )


    return probability_array


def mlp_predict_default_probability(
    X
):

    probabilities = (
        mlp_predict_proba_numpy(
            X
        )
    )


    return (
        probabilities[
            :,
            1
        ]
    )



# 18. Verify prediction wrapper


test_probability_output = (
    mlp_predict_proba_numpy(
        global_explain_np[
            :1
        ]
    )
)


print(
    "Test probability output:",
    test_probability_output
)


print(
    "Probability output shape:",
    test_probability_output.shape
)



# 19. Create Kernel SHAP explainer


shap_explainer = (
    shap.KernelExplainer(
        model=(
            mlp_predict_default_probability
        ),
        data=(
            background_summary
        ),
        feature_names=(
            feature_names
        ),
        link="identity"
    )
)


print(
    "Kernel SHAP explainer created."
)



# 20. Global SHAP explanation


global_shap_values = (
    shap_explainer
    .shap_values(
        global_explain_np,
        nsamples=SHAP_NSAMPLES,
        l1_reg=(
            f"num_features("
            f"{TOP_FEATURES}"
            f")"
        )
    )
)


if isinstance(
    global_shap_values,
    list
):

    global_shap_values = (
        global_shap_values[0]
    )


global_shap_values = (
    np.asarray(
        global_shap_values
    )
)


print(
    "Global SHAP values shape:",
    global_shap_values.shape
)



# 21. Calculate transformed-feature global SHAP importance


mean_absolute_shap = (

    np.abs(
        global_shap_values
    )

    .mean(
        axis=0
    )
)


global_shap_rows = [

    (
        feature_names[i],
        get_original_feature_group(
            feature_names[i]
        ),
        float(
            mean_absolute_shap[i]
        )
    )

    for i
    in range(
        len(
            feature_names
        )
    )
]


global_shap_df = (

    spark.createDataFrame(
        global_shap_rows,
        [
            "transformed_feature",
            "original_feature",
            "mean_absolute_shap"
        ]
    )

    .orderBy(
        desc(
            "mean_absolute_shap"
        )
    )
)


print(
    "Top MLP FICO transformed "
    "features according to SHAP:"
)


display(
    global_shap_df
    .limit(
        TOP_FEATURES
    )
)


global_shap_df.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_mlp_fico_shap_global"
    )


# 
# # 22. Aggregate SHAP back to original business variables
# 

# grouped_shap_df = (

#     global_shap_df

#     .groupBy(
#         "original_feature"
#     )

#     .sum(
#         "mean_absolute_shap"
#     )

#     .withColumnRenamed(
#         "sum(mean_absolute_shap)",
#         "grouped_mean_absolute_shap"
#     )

#     .orderBy(
#         desc(
#             "grouped_mean_absolute_shap"
#         )
#     )
# )


# print(
#     "Grouped MLP SHAP importance "
#     "at original-feature level:"
# )


# display(
#     grouped_shap_df
#     .limit(
#         TOP_FEATURES
#     )
# )


# grouped_shap_df.write \
#     .mode("overwrite") \
#     .option(
#         "overwriteSchema",
#         "true"
#     ) \
#     .format("delta") \
#     .saveAsTable(
#         "dissertation.lendingclub."
#         "lc_xai_mlp_fico_shap_global_grouped"
#     )


# 22. Aggregate SHAP to original business variables
#
# IMPORTANT:
# Group SHAP contributions within each observation FIRST,
# then calculate mean absolute group contribution.
#
# This avoids artificially favouring categorical variables
# simply because they have many one-hot encoded dimensions.


from collections import defaultdict


# Build:
# original feature -> positions in model feature vector

group_feature_indices = defaultdict(list)


for feature_index, transformed_feature in enumerate(
    feature_names
):

    original_feature = (
        get_original_feature_group(
            transformed_feature
        )
    )

    group_feature_indices[
        original_feature
    ].append(
        feature_index
    )


grouped_shap_rows = []


for (
    original_feature,
    feature_indices
) in group_feature_indices.items():


    # For every explained borrower:
    # sum SHAP contributions of all encoded dimensions
    # belonging to this original variable.


    grouped_contributions = (
        global_shap_values[
            :,
            feature_indices
        ]
        .sum(
            axis=1
        )
    )



    # Global importance of the original feature


    grouped_mean_absolute_shap = float(
        np.mean(
            np.abs(
                grouped_contributions
            )
        )
    )


    grouped_shap_rows.append(
        (
            original_feature,
            grouped_mean_absolute_shap,
            int(
                len(
                    feature_indices
                )
            )
        )
    )


grouped_shap_df = (

    spark.createDataFrame(
        grouped_shap_rows,
        [
            "original_feature",
            "grouped_mean_absolute_shap",
            "number_of_transformed_features"
        ]
    )

    .orderBy(
        desc(
            "grouped_mean_absolute_shap"
        )
    )
)


print(
    "Grouped MLP SHAP importance "
    "at original-feature level:"
)


display(
    grouped_shap_df
    .limit(
        TOP_FEATURES
    )
)


grouped_shap_df.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_mlp_fico_shap_global_grouped"
    )

# 23. Save global SHAP summary plot


plt.figure()


shap.summary_plot(
    global_shap_values,
    global_explain_np,
    feature_names=(
        feature_names
    ),
    max_display=(
        TOP_FEATURES
    ),
    show=False
)


plt.title(
    "MLP FICO - SHAP Summary"
)


plt.tight_layout()


shap_summary_path = (
    os.path.join(
        OUTPUT_DIR,
        "mlp_fico_shap_summary.png"
    )
)


plt.savefig(
    shap_summary_path,
    dpi=200,
    bbox_inches="tight"
)


plt.show()
plt.close()


print(
    "Saved:",
    shap_summary_path
)



# 24. Save global SHAP importance plot


plt.figure()


shap.summary_plot(
    global_shap_values,
    global_explain_np,
    feature_names=(
        feature_names
    ),
    plot_type="bar",
    max_display=(
        TOP_FEATURES
    ),
    show=False
)


plt.title(
    "MLP FICO - Global SHAP Importance"
)


plt.tight_layout()


global_plot_path = (
    os.path.join(
        OUTPUT_DIR,
        "mlp_fico_shap_global_importance.png"
    )
)


plt.savefig(
    global_plot_path,
    dpi=200,
    bbox_inches="tight"
)


plt.show()
plt.close()


print(
    "Saved:",
    global_plot_path
)



# 25. Plot grouped original-feature SHAP importance


grouped_top_pd = (

    grouped_shap_df

    .limit(
        TOP_FEATURES
    )

    .toPandas()
)


grouped_top_pd = (
    grouped_top_pd
    .iloc[::-1]
)


plt.figure(
    figsize=(
        9,
        8
    )
)


plt.barh(
    grouped_top_pd[
        "original_feature"
    ],

    grouped_top_pd[
        "grouped_mean_absolute_shap"
    ]
)


plt.xlabel(
    "Grouped mean(|SHAP value|)"
)


plt.ylabel(
    "Original feature"
)


plt.title(
    "MLP FICO - Grouped Global SHAP Importance"
)


plt.tight_layout()


grouped_plot_path = (
    os.path.join(
        OUTPUT_DIR,
        "mlp_fico_shap_global_grouped.png"
    )
)


plt.savefig(
    grouped_plot_path,
    dpi=200,
    bbox_inches="tight"
)


plt.show()
plt.close()


print(
    "Saved:",
    grouped_plot_path
)



# 26. Prepare local SHAP cases


local_np = (
    np.vstack(
        [
            case[
                "features"
            ].toArray()

            for case
            in cases
        ]
    )
)



# 27. Calculate local SHAP values


local_shap_values = (
    shap_explainer
    .shap_values(
        local_np,
        nsamples=SHAP_NSAMPLES,
        l1_reg=(
            f"num_features("
            f"{TOP_FEATURES}"
            f")"
        )
    )
)


if isinstance(
    local_shap_values,
    list
):

    local_shap_values = (
        local_shap_values[0]
    )


local_shap_values = (
    np.asarray(
        local_shap_values
    )
)


print(
    "Local SHAP values shape:",
    local_shap_values.shape
)



# 28. Helper for original human-readable feature values


# def get_original_value_info(
#     transformed_feature,
#     raw_values
# ):

#     (
#         original_feature,
#         encoded_category
#     ) = (
#         get_encoded_category_info(
#             transformed_feature
#         )
#     )


#     original_value = (
#         raw_values.get(
#             original_feature
#         )
#     )


#    
#     # Normal numerical / engineered variable
#    
#     sentinel_missing_features = {
#     "mths_since_last_delinq",
#     "emp_length_years",
#     "mths_since_rcnt_il",
#     "il_util"
#     }


#     if (
#         original_feature
#         in sentinel_missing_features
#         and original_value == -1
#     ):
#         return {
#         "original_feature":
#             original_feature,

#         "original_value":
#             original_value,

#         "encoded_category":
#             None,

#         "category_status":
#             None,

#         "display_value":
#             "Missing"
#     }

#     # if encoded_category is None:

#     #     return {

#     #         "original_feature":
#     #             original_feature,

#     #         "original_value":
#     #             original_value,

#     #         "encoded_category":
#     #             None,

#     #         "category_status":
#     #             None,

#     #         "display_value":
#     #             str(
#     #                 original_value
#     #             )
#     #     }


#    
#     # One-hot encoded categorical variable
#    

#     category_status = (

#         "active"

#         if str(
#             original_value
#         ) == str(
#             encoded_category
#         )

#         else "inactive"
#     )


#     display_value = (

#         f"{original_value} "
#         f"({category_status})"
#     )


#     return {

#         "original_feature":
#             original_feature,

#         "original_value":
#             original_value,

#         "encoded_category":
#             encoded_category,

#         "category_status":
#             category_status,

#         "display_value":
#             display_value
#     }

def get_original_value_info(
    transformed_feature,
    raw_values
):

    (
        original_feature,
        encoded_category
    ) = get_encoded_category_info(
        transformed_feature
    )

    original_value = raw_values.get(
        original_feature
    )


    # Numerical / engineered variable


    if encoded_category is None:

        sentinel_missing_features = {
            "mths_since_last_delinq",
            "emp_length_years",
            "mths_since_rcnt_il",
            "il_util"
        }

        # Show meaningful label instead of -1
        if (
            original_feature
            in sentinel_missing_features
            and original_value == -1
        ):

            display_value = "Missing"

        else:

            display_value = str(
                original_value
            )

        return {
            "original_feature":
                original_feature,

            "original_value":
                original_value,

            "encoded_category":
                None,

            "category_status":
                None,

            "display_value":
                display_value
        }



    # One-hot encoded categorical variable


    category_is_active = (
        str(original_value)
        == str(encoded_category)
    )

    category_status = (
        "active"
        if category_is_active
        else "inactive"
    )

    display_value = (
        f"{encoded_category} "
        f"({category_status})"
    )

    return {
        "original_feature":
            original_feature,

        "original_value":
            original_value,

        "encoded_category":
            encoded_category,

        "category_status":
            category_status,

        "display_value":
            display_value
    }

# 29. Create final readable local SHAP table


local_shap_rows = []


for (
    case_index,
    case
) in enumerate(
    cases
):


    shap_row = (
        local_shap_values[
            case_index
        ]
    )


    importance_order = (
        np.argsort(
            np.abs(
                shap_row
            )
        )[::-1]
    )


    for (
        rank,
        feature_index
    ) in enumerate(
        importance_order[
            :TOP_FEATURES
        ],
        start=1
    ):


        transformed_feature = (
            feature_names[
                feature_index
            ]
        )


        original_info = (
            get_original_value_info(
                transformed_feature,
                case[
                    "raw_values"
                ]
            )
        )


        original_value = (
            original_info[
                "original_value"
            ]
        )


        # Convert all original values to text for a stable
        # Spark schema across numeric/categorical variables.
        original_value_string = (

            None

            if original_value is None

            else str(
                original_value
            )
        )


        local_shap_rows.append(
            (
                case[
                    "case_type"
                ],

                case[
                    "actual"
                ],

                case[
                    "prediction"
                ],

                case[
                    "default_probability"
                ],

                int(
                    rank
                ),

                transformed_feature,

                original_info[
                    "original_feature"
                ],

                original_value_string,

                original_info[
                    "encoded_category"
                ],

                original_info[
                    "category_status"
                ],

                float(
                    local_np[
                        case_index,
                        feature_index
                    ]
                ),

                float(
                    shap_row[
                        feature_index
                    ]
                ),

                float(
                    abs(
                        shap_row[
                            feature_index
                        ]
                    )
                )
            )
        )


local_shap_df = (
    spark.createDataFrame(
        local_shap_rows,
        [
            "case_type",
            "actual_default_flag",
            "prediction",
            "default_probability",
            "rank",
            "transformed_feature",
            "original_feature",
            "original_value",
            "encoded_category",
            "category_status",
            "transformed_value",
            "shap_value",
            "absolute_shap"
        ]
    )
)


print(
    "Readable local SHAP explanations:"
)


display(
    local_shap_df
    .orderBy(
        "case_type",
        "rank"
    )
)


local_shap_df.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_mlp_fico_shap_local"
    )



# 30. Prepare human-readable waterfall display values


def build_waterfall_display_values(
    case
):

    display_values = []


    for transformed_feature in (
        feature_names
    ):

        feature_info = (
            get_original_value_info(
                transformed_feature,
                case[
                    "raw_values"
                ]
            )
        )


        display_values.append(
            feature_info[
                "display_value"
            ]
        )


    return np.asarray(
        display_values,
        dtype=object
    )



# 31. Determine Kernel SHAP expected value


expected_value = float(

    np.asarray(
        shap_explainer.expected_value
    )

    .reshape(
        -1
    )[0]
)


print(
    "SHAP expected default probability:",
    expected_value
)


print("\nSHAP additivity check:")

for case_index, case in enumerate(cases):

    reconstructed_probability = (
        expected_value
        + local_shap_values[case_index].sum()
    )

    actual_probability = (
        case["default_probability"]
    )

    difference = abs(
        reconstructed_probability
        - actual_probability
    )

    print(
        case["case_type"],
        "| model probability:",
        round(actual_probability, 6),
        "| SHAP reconstructed:",
        round(reconstructed_probability, 6),
        "| difference:",
        round(difference, 8)
    )
# 32. Create final local waterfall plots


for (
    case_index,
    case
) in enumerate(
    cases
):


    display_values = (
        build_waterfall_display_values(
            case
        )
    )


    explanation = (
        shap.Explanation(

            values=(
                local_shap_values[
                    case_index
                ]
            ),

            base_values=(
                expected_value
            ),

            data=(
                local_np[
                    case_index
                ]
            ),

            display_data=(
                display_values
            ),

            feature_names=(
                feature_names
            )
        )
    )


    shap.plots.waterfall(
        explanation,
        max_display=(
            TOP_FEATURES
        ),
        show=False
    )


    plt.title(
        (
            "MLP FICO - SHAP "
            f"{case['case_type']}"
        )
    )


    plt.tight_layout()


    local_plot_path = (
        os.path.join(
            OUTPUT_DIR,
            (
                "mlp_fico_shap_"
                f"{case['case_type']}.png"
            )
        )
    )


    plt.savefig(
        local_plot_path,
        dpi=200,
        bbox_inches="tight"
    )


    plt.show()
    plt.close()


    print(
        "Saved:",
        local_plot_path
    )



# 33. Save SHAP methodology/configuration


config_rows = [

    (
        int(
            BACKGROUND_POOL_ROWS
        ),

        int(
            BACKGROUND_CLUSTERS
        ),

        int(
            GLOBAL_EXPLAIN_ROWS
        ),

        int(
            SHAP_NSAMPLES
        ),

        int(
            TOP_FEATURES
        ),

        int(
            RANDOM_SEED
        ),

        int(
            mlp_model.numFeatures
        )
    )
]


config_df = (
    spark.createDataFrame(
        config_rows,
        [
            "background_pool_rows",
            "background_clusters",
            "global_explain_rows",
            "shap_nsamples",
            "top_features",
            "random_seed",
            "mlp_input_features"
        ]
    )
)


config_df.write \
    .mode("overwrite") \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format("delta") \
    .saveAsTable(
        "dissertation.lendingclub."
        "lc_xai_mlp_fico_shap_config"
    )



# 34. Final output summary


print(
    "\n"
    "09_posthoc_xai.py completed successfully."
)


print(
    "\nCreated Delta tables:"
)


print(
    "1. dissertation.lendingclub."
    "lc_xai_mlp_fico_local_cases"
)


print(
    "2. dissertation.lendingclub."
    "lc_xai_mlp_fico_shap_global"
)


print(
    "3. dissertation.lendingclub."
    "lc_xai_mlp_fico_shap_global_grouped"
)


print(
    "4. dissertation.lendingclub."
    "lc_xai_mlp_fico_shap_local"
)


print(
    "5. dissertation.lendingclub."
    "lc_xai_mlp_fico_shap_config"
)


print(
    "\nFigures saved in:"
)


print(
    OUTPUT_DIR
)


print(
    "\nFinal XAI configuration:"
)


print(
    "Background pool:",
    BACKGROUND_POOL_ROWS
)


print(
    "Background k-means clusters:",
    BACKGROUND_CLUSTERS
)


print(
    "Global explanation sample:",
    GLOBAL_EXPLAIN_ROWS
)


print(
    "Kernel SHAP nsamples:",
    SHAP_NSAMPLES
)


print(
    "MLP features:",
    mlp_model.numFeatures
)