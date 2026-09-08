# ============================================================
# 11_final_technical_validation.py
#
# Final technical QA / reproducibility audit
# for the dissertation ML + XAI pipeline.
#
# This script:
# - does NOT modify the modelling datasets
# - does NOT retrain models
# - does NOT alter saved predictions or metrics
#
# The FICO-enriched dataset intentionally contains:
#   fico_score
#   fico_score_missing
#   pub_rec_bankruptcies
#   pub_rec_bankruptcies_missing
#   has_bankruptcy
#   mort_acc
#   mort_acc_missing
#
# Therefore, the FICO experiment should be described in the
# dissertation as a FICO / credit-profile enrichment experiment,
# rather than as the addition of fico_score alone.
# ============================================================


from pyspark.sql import functions as F


# ============================================================
# 1. Configuration
# ============================================================

NO_FICO_TABLE = (
    "dissertation.lendingclub."
    "lc_2007_2017_ml_no_leakage"
)

FICO_TABLE = (
    "dissertation.lendingclub."
    "lc_2007_2017_ml_no_leakage_fico"
)


EXPECTED_ROWS = 455318
EXPECTED_TEST_ROWS = 90884
EXPECTED_FEATURE_VECTOR_SIZE = 135

RANDOM_SEED = 42
TOP_FEATURES = 20


# Variables intentionally added to the enriched FICO dataset
EXPECTED_ENRICHMENT_ONLY_FEATURES = {
    "fico_score",
    "fico_score_missing",
    "pub_rec_bankruptcies",
    "pub_rec_bankruptcies_missing",
    "has_bankruptcy",
    "mort_acc",
    "mort_acc_missing"
}


# Variables that must not be present in the final modelling
# feature tables because they represent leakage or
# LendingClub-derived risk assessments.
LEAKAGE_COLUMNS = {
    "grade",
    "sub_grade",
    "int_rate",
    "loan_status",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d"
}


# ============================================================
# 2. Validation result collector
# ============================================================

validation_results = []


def record_check(
    section,
    check_name,
    passed,
    details=""
):

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    validation_results.append(
        (
            section,
            check_name,
            status,
            str(details)
        )
    )

    print(
        f"[{status}] "
        f"{section} | "
        f"{check_name} | "
        f"{details}"
    )


# ============================================================
# 3. Dataset existence
# ============================================================

print(
    "\n"
    "============================================================"
)
print("1. DATASET EXISTENCE")
print(
    "============================================================"
)


for table_name in [
    NO_FICO_TABLE,
    FICO_TABLE
]:

    record_check(
        "Dataset",
        f"Table exists: {table_name}",
        spark.catalog.tableExists(
            table_name
        )
    )


if not spark.catalog.tableExists(
    NO_FICO_TABLE
):

    raise RuntimeError(
        f"Missing required table: {NO_FICO_TABLE}"
    )


if not spark.catalog.tableExists(
    FICO_TABLE
):

    raise RuntimeError(
        f"Missing required table: {FICO_TABLE}"
    )


df_no_fico = spark.table(
    NO_FICO_TABLE
)

df_fico = spark.table(
    FICO_TABLE
)


# ============================================================
# 4. Row counts
# ============================================================

print(
    "\n"
    "============================================================"
)
print("2. ROW COUNTS")
print(
    "============================================================"
)


no_fico_count = (
    df_no_fico.count()
)

fico_count = (
    df_fico.count()
)


record_check(
    "Dataset",
    "No-FICO row count",
    no_fico_count == EXPECTED_ROWS,
    f"{no_fico_count:,}"
)


record_check(
    "Dataset",
    "FICO-enriched row count",
    fico_count == EXPECTED_ROWS,
    f"{fico_count:,}"
)


record_check(
    "Dataset",
    "Dataset row counts are identical",
    no_fico_count == fico_count,
    (
        f"no-FICO={no_fico_count:,}, "
        f"FICO-enriched={fico_count:,}"
    )
)


# ============================================================
# 5. Identifier handling
# ============================================================

print(
    "\n"
    "============================================================"
)
print("3. IDENTIFIER HANDLING")
print(
    "============================================================"
)


no_fico_has_id = (
    "id" in df_no_fico.columns
)

fico_has_id = (
    "id" in df_fico.columns
)


record_check(
    "Identifier",
    "No-FICO identifier handling",
    True,
    (
        "id retained"
        if no_fico_has_id
        else
        "id intentionally absent from final ML feature table; "
        "row fingerprints used for validation"
    )
)


record_check(
    "Identifier",
    "FICO-enriched identifier handling",
    True,
    (
        "id retained"
        if fico_has_id
        else
        "id intentionally absent from final ML feature table; "
        "row fingerprints used for validation"
    )
)


if no_fico_has_id:

    no_fico_null_ids = (
        df_no_fico
        .filter(
            F.col("id").isNull()
        )
        .count()
    )

    no_fico_duplicate_ids = (
        df_no_fico
        .groupBy("id")
        .count()
        .filter(
            F.col("count") > 1
        )
        .count()
    )

    record_check(
        "Identifier",
        "No-FICO IDs contain no NULLs",
        no_fico_null_ids == 0,
        f"NULL IDs={no_fico_null_ids}"
    )

    record_check(
        "Identifier",
        "No-FICO IDs are unique",
        no_fico_duplicate_ids == 0,
        f"duplicate IDs={no_fico_duplicate_ids}"
    )


if fico_has_id:

    fico_null_ids = (
        df_fico
        .filter(
            F.col("id").isNull()
        )
        .count()
    )

    fico_duplicate_ids = (
        df_fico
        .groupBy("id")
        .count()
        .filter(
            F.col("count") > 1
        )
        .count()
    )

    record_check(
        "Identifier",
        "FICO-enriched IDs contain no NULLs",
        fico_null_ids == 0,
        f"NULL IDs={fico_null_ids}"
    )

    record_check(
        "Identifier",
        "FICO-enriched IDs are unique",
        fico_duplicate_ids == 0,
        f"duplicate IDs={fico_duplicate_ids}"
    )


# ============================================================
# 6. Target validation
# ============================================================

print(
    "\n"
    "============================================================"
)
print("4. TARGET VARIABLE")
print(
    "============================================================"
)


for (
    dataset_name,
    dataframe
) in [

    (
        "No-FICO",
        df_no_fico
    ),

    (
        "FICO-enriched",
        df_fico
    )

]:

    target_nulls = (
        dataframe
        .filter(
            F.col(
                "default_flag"
            ).isNull()
        )
        .count()
    )


    target_values = {

        int(
            row[
                "default_flag"
            ]
        )

        for row in (

            dataframe

            .select(
                "default_flag"
            )

            .distinct()

            .collect()
        )

        if row[
            "default_flag"
        ] is not None
    }


    record_check(
        "Target",
        f"{dataset_name}: no NULL target values",
        target_nulls == 0,
        f"NULL targets={target_nulls}"
    )


    record_check(
        "Target",
        f"{dataset_name}: default_flag is binary",
        target_values == {0, 1},
        f"values={sorted(target_values)}"
    )


print(
    "\nNo-FICO target distribution:"
)

df_no_fico \
    .groupBy(
        "default_flag"
    ) \
    .count() \
    .orderBy(
        "default_flag"
    ) \
    .show()


print(
    "\nFICO-enriched target distribution:"
)

df_fico \
    .groupBy(
        "default_flag"
    ) \
    .count() \
    .orderBy(
        "default_flag"
    ) \
    .show()


# ============================================================
# 7. Leakage validation
# ============================================================

print(
    "\n"
    "============================================================"
)
print("5. DATA LEAKAGE CHECK")
print(
    "============================================================"
)


for (
    dataset_name,
    dataframe
) in [

    (
        "No-FICO",
        df_no_fico
    ),

    (
        "FICO-enriched",
        df_fico
    )

]:

    present_leakage = sorted(
        LEAKAGE_COLUMNS.intersection(
            set(
                dataframe.columns
            )
        )
    )


    record_check(
        "Leakage",
        f"{dataset_name}: prohibited leakage variables absent",
        len(
            present_leakage
        ) == 0,
        (
            "none"
            if len(
                present_leakage
            ) == 0
            else present_leakage
        )
    )


# ============================================================
# 8. FICO / credit-profile enrichment validation
# ============================================================

print(
    "\n"
    "============================================================"
)
print("6. FICO / CREDIT-PROFILE ENRICHMENT")
print(
    "============================================================"
)


ignored_columns = {
    "id",
    "default_flag"
}


no_fico_features = (
    set(
        df_no_fico.columns
    )
    -
    ignored_columns
)


fico_features = (
    set(
        df_fico.columns
    )
    -
    ignored_columns
)


extra_in_fico = sorted(
    fico_features
    -
    no_fico_features
)


extra_in_no_fico = sorted(
    no_fico_features
    -
    fico_features
)


print(
    "\nFeatures only in FICO-enriched dataset:"
)

print(
    extra_in_fico
)


print(
    "\nFeatures only in no-FICO dataset:"
)

print(
    extra_in_no_fico
)


record_check(
    "Feature Alignment",
    "No unexpected features exist only in no-FICO",
    len(
        extra_in_no_fico
    ) == 0,
    extra_in_no_fico
)


unexpected_enrichment_features = (
    set(
        extra_in_fico
    )
    -
    EXPECTED_ENRICHMENT_ONLY_FEATURES
)


missing_expected_enrichment_features = (
    EXPECTED_ENRICHMENT_ONLY_FEATURES
    -
    set(
        extra_in_fico
    )
)


record_check(
    "Feature Alignment",
    (
        "FICO-enriched dataset contains only "
        "the intended enrichment variables"
    ),
    (
        len(
            unexpected_enrichment_features
        ) == 0
        and
        len(
            missing_expected_enrichment_features
        ) == 0
    ),
    (
        f"enrichment features={extra_in_fico}; "
        f"unexpected="
        f"{sorted(unexpected_enrichment_features)}; "
        f"missing expected="
        f"{sorted(missing_expected_enrichment_features)}"
    )
)


record_check(
    "FICO",
    "fico_score present in enriched dataset",
    "fico_score"
    in
    df_fico.columns
)


record_check(
    "FICO",
    "fico_score absent from no-FICO dataset",
    "fico_score"
    not in
    df_no_fico.columns
)


for column_name in [
    "fico_range_low",
    "fico_range_high"
]:

    record_check(
        "FICO",
        (
            f"{column_name} excluded from "
            "final modelling table"
        ),
        column_name
        not in
        df_fico.columns
    )


# ============================================================
# 9. Common-row alignment using fingerprints
# ============================================================

print(
    "\n"
    "============================================================"
)
print("7. DATASET ROW ALIGNMENT")
print(
    "============================================================"
)


common_columns = sorted(
    (
        set(
            df_no_fico.columns
        )
        &
        set(
            df_fico.columns
        )
    )
    -
    {"id"}
)


print(
    "Number of shared columns used "
    "for row fingerprint:",
    len(
        common_columns
    )
)


def add_row_fingerprint(
    dataframe,
    columns
):

    return (

        dataframe

        .withColumn(
            "_row_fingerprint",

            F.sha2(
                F.to_json(
                    F.struct(
                        *[
                            F.col(
                                column_name
                            )

                            for column_name
                            in columns
                        ]
                    ),

                    options={
                        "ignoreNullFields":
                        "false"
                    }
                ),

                256
            )
        )
    )


no_fico_fingerprinted = (
    add_row_fingerprint(
        df_no_fico,
        common_columns
    )
)


fico_fingerprinted = (
    add_row_fingerprint(
        df_fico,
        common_columns
    )
)


no_fico_fingerprint_counts = (

    no_fico_fingerprinted

    .groupBy(
        "_row_fingerprint"
    )

    .count()

    .withColumnRenamed(
        "count",
        "no_fico_count"
    )
)


fico_fingerprint_counts = (

    fico_fingerprinted

    .groupBy(
        "_row_fingerprint"
    )

    .count()

    .withColumnRenamed(
        "count",
        "fico_count"
    )
)


fingerprint_comparison = (

    no_fico_fingerprint_counts

    .join(
        fico_fingerprint_counts,
        on="_row_fingerprint",
        how="full_outer"
    )

    .fillna(
        0,
        subset=[
            "no_fico_count",
            "fico_count"
        ]
    )
)


mismatched_fingerprints = (

    fingerprint_comparison

    .filter(
        F.col(
            "no_fico_count"
        )
        !=
        F.col(
            "fico_count"
        )
    )

    .count()
)


record_check(
    "Dataset Alignment",
    (
        "FICO-enriched and no-FICO datasets "
        "contain identical shared-variable rows"
    ),
    mismatched_fingerprints == 0,
    (
        "mismatched fingerprints="
        f"{mismatched_fingerprints}"
    )
)


# ============================================================
# 10. Missing-value validation
# ============================================================

print(
    "\n"
    "============================================================"
)
print("8. NULL VALUE CHECK")
print(
    "============================================================"
)


def validate_missing_values(
    dataframe,
    dataset_name
):

    predictor_columns = [

        column_name

        for column_name
        in dataframe.columns

        if column_name
        not in {
            "id",
            "default_flag"
        }
    ]


    null_counts_row = (

        dataframe

        .select(
            [
                F.sum(
                    F.when(
                        F.col(
                            column_name
                        ).isNull(),
                        1
                    )
                    .otherwise(
                        0
                    )
                )
                .alias(
                    column_name
                )

                for column_name
                in predictor_columns
            ]
        )

        .first()
    )


    columns_with_nulls = {

        column_name:
        int(
            null_counts_row[
                column_name
            ]
        )

        for column_name
        in predictor_columns

        if (
            null_counts_row[
                column_name
            ]
            is not None
            and
            null_counts_row[
                column_name
            ] > 0
        )
    }


    record_check(
        "Missing Values",
        f"{dataset_name}: no NULL predictors",
        len(
            columns_with_nulls
        ) == 0,
        (
            "none"
            if len(
                columns_with_nulls
            ) == 0
            else columns_with_nulls
        )
    )


validate_missing_values(
    df_no_fico,
    "No-FICO"
)


validate_missing_values(
    df_fico,
    "FICO-enriched"
)


# ============================================================
# 11. NaN validation
# ============================================================

print(
    "\n"
    "============================================================"
)
print("9. NaN CHECK")
print(
    "============================================================"
)


def validate_nan_values(
    dataframe,
    dataset_name
):

    floating_columns = [

        column_name

        for (
            column_name,
            data_type
        )
        in dataframe.dtypes

        if (
            data_type
            in [
                "float",
                "double"
            ]
            and
            column_name
            !=
            "default_flag"
        )
    ]


    if len(
        floating_columns
    ) == 0:

        record_check(
            "NaN",
            f"{dataset_name}: no NaN values",
            True,
            "no floating-point predictors"
        )

        return


    nan_counts_row = (

        dataframe

        .select(
            [
                F.sum(
                    F.when(
                        F.isnan(
                            F.col(
                                column_name
                            )
                        ),
                        1
                    )
                    .otherwise(
                        0
                    )
                )
                .alias(
                    column_name
                )

                for column_name
                in floating_columns
            ]
        )

        .first()
    )


    columns_with_nan = {

        column_name:
        int(
            nan_counts_row[
                column_name
            ]
        )

        for column_name
        in floating_columns

        if (
            nan_counts_row[
                column_name
            ]
            is not None
            and
            nan_counts_row[
                column_name
            ] > 0
        )
    }


    record_check(
        "NaN",
        f"{dataset_name}: no NaN predictors",
        len(
            columns_with_nan
        ) == 0,
        (
            "none"
            if len(
                columns_with_nan
            ) == 0
            else columns_with_nan
        )
    )


validate_nan_values(
    df_no_fico,
    "No-FICO"
)


validate_nan_values(
    df_fico,
    "FICO-enriched"
)


# ============================================================
# 12. Recreate final train/test splits
# ============================================================

print(
    "\n"
    "============================================================"
)
print("10. TRAIN / TEST SPLIT")
print(
    "============================================================"
)


no_fico_train, no_fico_test = (

    df_no_fico

    .randomSplit(
        [
            0.8,
            0.2
        ],
        seed=RANDOM_SEED
    )
)


fico_train, fico_test = (

    df_fico

    .randomSplit(
        [
            0.8,
            0.2
        ],
        seed=RANDOM_SEED
    )
)


no_fico_train_count = (
    no_fico_train.count()
)

no_fico_test_count = (
    no_fico_test.count()
)

fico_train_count = (
    fico_train.count()
)

fico_test_count = (
    fico_test.count()
)


print(
    "No-FICO train:",
    no_fico_train_count
)

print(
    "No-FICO test:",
    no_fico_test_count
)

print(
    "FICO-enriched train:",
    fico_train_count
)

print(
    "FICO-enriched test:",
    fico_test_count
)


record_check(
    "Split",
    "No-FICO split reconstructs all rows",
    (
        no_fico_train_count
        +
        no_fico_test_count
    )
    ==
    no_fico_count
)


record_check(
    "Split",
    "FICO-enriched split reconstructs all rows",
    (
        fico_train_count
        +
        fico_test_count
    )
    ==
    fico_count
)


record_check(
    "Split",
    "No-FICO test size matches final modelling run",
    no_fico_test_count
    ==
    EXPECTED_TEST_ROWS,
    f"{no_fico_test_count:,}"
)


record_check(
    "Split",
    "FICO-enriched test size matches final modelling run",
    fico_test_count
    ==
    EXPECTED_TEST_ROWS,
    f"{fico_test_count:,}"
)


# ============================================================
# 13. Test-set alignment using fingerprints
# ============================================================

print(
    "\n"
    "============================================================"
)
print("11. TEST-SPLIT ALIGNMENT")
print(
    "============================================================"
)


no_fico_test_hashed = (
    add_row_fingerprint(
        no_fico_test,
        common_columns
    )
)


fico_test_hashed = (
    add_row_fingerprint(
        fico_test,
        common_columns
    )
)


no_fico_test_counts = (

    no_fico_test_hashed

    .groupBy(
        "_row_fingerprint"
    )

    .count()

    .withColumnRenamed(
        "count",
        "no_fico_test_count"
    )
)


fico_test_counts = (

    fico_test_hashed

    .groupBy(
        "_row_fingerprint"
    )

    .count()

    .withColumnRenamed(
        "count",
        "fico_test_count"
    )
)


test_fingerprint_comparison = (

    no_fico_test_counts

    .join(
        fico_test_counts,
        on="_row_fingerprint",
        how="full_outer"
    )

    .fillna(
        0,
        subset=[
            "no_fico_test_count",
            "fico_test_count"
        ]
    )
)


test_split_mismatches = (

    test_fingerprint_comparison

    .filter(
        F.col(
            "no_fico_test_count"
        )
        !=
        F.col(
            "fico_test_count"
        )
    )

    .count()
)


record_check(
    "Split",
    (
        "FICO-enriched and no-FICO "
        "test observations align"
    ),
    test_split_mismatches == 0,
    (
        "mismatched test fingerprints="
        f"{test_split_mismatches}"
    )
)


# ============================================================
# 14. Final model metrics tables
# ============================================================

print(
    "\n"
    "============================================================"
)
print("12. MODEL METRICS TABLES")
print(
    "============================================================"
)


METRIC_TABLES = {

    "LR no-FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_logistic_regression_metrics"
        ),

    "DT no-FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_decision_tree_metrics"
        ),

    "RF no-FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_random_forest_metrics"
        ),

    "MLP no-FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_deep_learning_mlp_metrics"
        ),

    "LR FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_logistic_regression_fico_metrics"
        ),

    "DT FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_decision_tree_fico_metrics"
        ),

    "RF FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_random_forest_fico_metrics"
        ),

    "MLP FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_deep_learning_mlp_fico_metrics"
        )
}


for (
    model_name,
    table_name
) in METRIC_TABLES.items():

    exists = (
        spark.catalog.tableExists(
            table_name
        )
    )

    record_check(
        "Metrics",
        f"{model_name}: table exists",
        exists,
        table_name
    )


    if exists:

        metric_rows = (
            spark.table(
                table_name
            )
            .count()
        )

        record_check(
            "Metrics",
            f"{model_name}: exactly one metrics row",
            metric_rows == 1,
            f"rows={metric_rows}"
        )


# ============================================================
# 15. Prediction tables
# ============================================================

print(
    "\n"
    "============================================================"
)
print("13. PREDICTION TABLES")
print(
    "============================================================"
)


PREDICTION_TABLES = {

    "LR no-FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_logistic_regression_predictions"
        ),

    "DT no-FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_decision_tree_predictions"
        ),

    "RF no-FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_random_forest_predictions"
        ),

    "MLP no-FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_deep_learning_mlp_predictions"
        ),

    "LR FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_logistic_regression_fico_predictions"
        ),

    "DT FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_decision_tree_fico_predictions"
        ),

    "RF FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_random_forest_fico_predictions"
        ),

    "MLP FICO":
        (
            "dissertation.lendingclub."
            "lc_2007_2017_deep_learning_mlp_fico_predictions"
        )
}


for (
    model_name,
    table_name
) in PREDICTION_TABLES.items():

    exists = (
        spark.catalog.tableExists(
            table_name
        )
    )


    record_check(
        "Predictions",
        f"{model_name}: prediction table exists",
        exists,
        table_name
    )


    if exists:

        prediction_count = (
            spark.table(
                table_name
            )
            .count()
        )


        record_check(
            "Predictions",
            (
                f"{model_name}: prediction count "
                "matches test set"
            ),
            prediction_count
            ==
            EXPECTED_TEST_ROWS,
            f"{prediction_count:,}"
        )


# ============================================================
# 16. Metric reproducibility from confusion matrices
# ============================================================

print(
    "\n"
    "============================================================"
)
print("14. METRIC REPRODUCIBILITY")
print(
    "============================================================"
)


TOLERANCE = 1e-8


def safe_divide(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        /
        denominator
    )


for model_name in (
    PREDICTION_TABLES.keys()
):

    prediction_table = (
        PREDICTION_TABLES[
            model_name
        ]
    )

    metrics_table = (
        METRIC_TABLES[
            model_name
        ]
    )


    if not (
        spark.catalog.tableExists(
            prediction_table
        )
        and
        spark.catalog.tableExists(
            metrics_table
        )
    ):

        continue


    predictions = spark.table(
        prediction_table
    )


    confusion_rows = (

        predictions

        .groupBy(
            "default_flag",
            "prediction"
        )

        .count()

        .collect()
    )


    confusion = {

        (
            int(
                row[
                    "default_flag"
                ]
            ),
            int(
                row[
                    "prediction"
                ]
            )
        ):
        int(
            row[
                "count"
            ]
        )

        for row
        in confusion_rows
    }


    tn = confusion.get(
        (0, 0),
        0
    )

    fp = confusion.get(
        (0, 1),
        0
    )

    fn = confusion.get(
        (1, 0),
        0
    )

    tp = confusion.get(
        (1, 1),
        0
    )


    total = (
        tn
        +
        fp
        +
        fn
        +
        tp
    )


    calculated_accuracy = (
        safe_divide(
            tp + tn,
            total
        )
    )


    calculated_default_precision = (
        safe_divide(
            tp,
            tp + fp
        )
    )


    calculated_default_recall = (
        safe_divide(
            tp,
            tp + fn
        )
    )


    calculated_default_f1 = (
        safe_divide(
            2
            *
            calculated_default_precision
            *
            calculated_default_recall,

            calculated_default_precision
            +
            calculated_default_recall
        )
    )


    stored = (
        spark.table(
            metrics_table
        )
        .first()
    )


    metric_checks = {

        "accuracy":
            (
                calculated_accuracy,
                float(
                    stored[
                        "accuracy"
                    ]
                )
            ),

        "default_precision":
            (
                calculated_default_precision,
                float(
                    stored[
                        "default_precision"
                    ]
                )
            ),

        "default_recall":
            (
                calculated_default_recall,
                float(
                    stored[
                        "default_recall"
                    ]
                )
            ),

        "default_f1":
            (
                calculated_default_f1,
                float(
                    stored[
                        "default_f1"
                    ]
                )
            )
    }


    for (
        metric_name,
        (
            calculated_value,
            stored_value
        )
    ) in metric_checks.items():

        difference = abs(
            calculated_value
            -
            stored_value
        )


        record_check(
            "Metric Reproduction",
            (
                f"{model_name}: "
                f"{metric_name}"
            ),
            difference
            <=
            TOLERANCE,
            (
                f"calculated="
                f"{calculated_value:.10f}, "
                f"stored="
                f"{stored_value:.10f}"
            )
        )


# ============================================================
# 17. Native XAI tables from file 08
# ============================================================

print(
    "\n"
    "============================================================"
)
print("15. NATIVE XAI OUTPUTS")
print(
    "============================================================"
)


XAI_08_TABLES = {

    (
        "dissertation.lendingclub."
        "lc_xai_logistic_regression_fico_coefficients"
    ):
        EXPECTED_FEATURE_VECTOR_SIZE,

    (
        "dissertation.lendingclub."
        "lc_xai_decision_tree_fico_importance"
    ):
        EXPECTED_FEATURE_VECTOR_SIZE,

    (
        "dissertation.lendingclub."
        "lc_xai_random_forest_fico_importance"
    ):
        EXPECTED_FEATURE_VECTOR_SIZE
}


for (
    table_name,
    expected_rows
) in XAI_08_TABLES.items():

    exists = (
        spark.catalog.tableExists(
            table_name
        )
    )


    record_check(
        "XAI 08",
        f"Table exists: {table_name}",
        exists
    )


    if exists:

        actual_rows = (
            spark.table(
                table_name
            )
            .count()
        )


        record_check(
            "XAI 08",
            f"Expected feature count: {table_name}",
            actual_rows
            ==
            expected_rows,
            (
                f"actual={actual_rows}, "
                f"expected={expected_rows}"
            )
        )


# ============================================================
# 18. Post-hoc SHAP outputs from file 09
# ============================================================

print(
    "\n"
    "============================================================"
)
print("16. POST-HOC SHAP OUTPUTS")
print(
    "============================================================"
)


XAI_09_EXPECTED_COUNTS = {

    (
        "dissertation.lendingclub."
        "lc_xai_mlp_fico_local_cases"
    ):
        4,

    (
        "dissertation.lendingclub."
        "lc_xai_mlp_fico_shap_global"
    ):
        EXPECTED_FEATURE_VECTOR_SIZE,

    (
        "dissertation.lendingclub."
        "lc_xai_mlp_fico_shap_local"
    ):
        4
        *
        TOP_FEATURES,

    (
        "dissertation.lendingclub."
        "lc_xai_mlp_fico_shap_config"
    ):
        1
}


for (
    table_name,
    expected_rows
) in XAI_09_EXPECTED_COUNTS.items():

    exists = (
        spark.catalog.tableExists(
            table_name
        )
    )


    record_check(
        "XAI 09",
        f"Table exists: {table_name}",
        exists
    )


    if exists:

        actual_rows = (
            spark.table(
                table_name
            )
            .count()
        )


        record_check(
            "XAI 09",
            f"Expected row count: {table_name}",
            actual_rows
            ==
            expected_rows,
            (
                f"actual={actual_rows}, "
                f"expected={expected_rows}"
            )
        )


GROUPED_SHAP_TABLE = (
    "dissertation.lendingclub."
    "lc_xai_mlp_fico_shap_global_grouped"
)


grouped_exists = (
    spark.catalog.tableExists(
        GROUPED_SHAP_TABLE
    )
)


record_check(
    "XAI 09",
    "Grouped SHAP table exists",
    grouped_exists
)


if grouped_exists:

    grouped_rows = (
        spark.table(
            GROUPED_SHAP_TABLE
        )
        .count()
    )


    record_check(
        "XAI 09",
        "Grouped SHAP table is non-empty",
        grouped_rows > 0,
        f"rows={grouped_rows}"
    )


# ============================================================
# 19. Local SHAP case validation
# ============================================================

LOCAL_CASE_TABLE = (
    "dissertation.lendingclub."
    "lc_xai_mlp_fico_local_cases"
)


if spark.catalog.tableExists(
    LOCAL_CASE_TABLE
):

    actual_case_types = {

        row[
            "case_type"
        ]

        for row in (

            spark.table(
                LOCAL_CASE_TABLE
            )

            .select(
                "case_type"
            )

            .distinct()

            .collect()
        )
    }


    expected_case_types = {
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative"
    }


    record_check(
        "XAI 09",
        "TP / TN / FP / FN cases all exist",
        actual_case_types
        ==
        expected_case_types,
        sorted(
            actual_case_types
        )
    )


# ============================================================
# 20. SHAP configuration validation
# ============================================================

SHAP_CONFIG_TABLE = (
    "dissertation.lendingclub."
    "lc_xai_mlp_fico_shap_config"
)


if spark.catalog.tableExists(
    SHAP_CONFIG_TABLE
):

    shap_config = (
        spark.table(
            SHAP_CONFIG_TABLE
        )
        .first()
    )


    record_check(
        "SHAP",
        "MLP input feature count",
        int(
            shap_config[
                "mlp_input_features"
            ]
        )
        ==
        EXPECTED_FEATURE_VECTOR_SIZE,
        (
            f"{shap_config['mlp_input_features']}"
        )
    )


    record_check(
        "SHAP",
        "Global explanation sample size",
        int(
            shap_config[
                "global_explain_rows"
            ]
        )
        ==
        100,
        (
            f"{shap_config['global_explain_rows']}"
        )
    )


    record_check(
        "SHAP",
        "Kernel SHAP nsamples",
        int(
            shap_config[
                "shap_nsamples"
            ]
        )
        ==
        200,
        (
            f"{shap_config['shap_nsamples']}"
        )
    )


    record_check(
        "SHAP",
        "Background pool size",
        int(
            shap_config[
                "background_pool_rows"
            ]
        )
        ==
        500,
        (
            f"{shap_config['background_pool_rows']}"
        )
    )


    record_check(
        "SHAP",
        "Background k-means clusters",
        int(
            shap_config[
                "background_clusters"
            ]
        )
        ==
        10,
        (
            f"{shap_config['background_clusters']}"
        )
    )


# ============================================================
# 21. Cross-model XAI evaluation from file 10
# ============================================================

print(
    "\n"
    "============================================================"
)
print("17. CROSS-MODEL XAI OUTPUTS")
print(
    "============================================================"
)


XAI_10_EXPECTED_COUNTS = {

    (
        "dissertation.lendingclub."
        "lc_xai_fico_all_model_importance"
    ):
        4
        *
        EXPECTED_FEATURE_VECTOR_SIZE,

    (
        "dissertation.lendingclub."
        "lc_xai_fico_feature_comparison"
    ):
        EXPECTED_FEATURE_VECTOR_SIZE,

    (
        "dissertation.lendingclub."
        "lc_xai_fico_pairwise_jaccard"
    ):
        12,

    (
        "dissertation.lendingclub."
        "lc_xai_fico_rank_correlation"
    ):
        6,

    (
        "dissertation.lendingclub."
        "lc_xai_fico_performance_interpretability"
    ):
        4,

    (
        "dissertation.lendingclub."
        "lc_xai_fico_evaluation_config"
    ):
        1
}


for (
    table_name,
    expected_rows
) in XAI_10_EXPECTED_COUNTS.items():

    exists = (
        spark.catalog.tableExists(
            table_name
        )
    )


    record_check(
        "XAI 10",
        f"Table exists: {table_name}",
        exists
    )


    if exists:

        actual_rows = (
            spark.table(
                table_name
            )
            .count()
        )


        record_check(
            "XAI 10",
            f"Expected row count: {table_name}",
            actual_rows
            ==
            expected_rows,
            (
                f"actual={actual_rows}, "
                f"expected={expected_rows}"
            )
        )


for table_name in [

    (
        "dissertation.lendingclub."
        "lc_xai_fico_consensus_features"
    ),

    (
        "dissertation.lendingclub."
        "lc_xai_fico_original_feature_consensus"
    )

]:

    exists = (
        spark.catalog.tableExists(
            table_name
        )
    )


    record_check(
        "XAI 10",
        f"Table exists: {table_name}",
        exists
    )


    if exists:

        actual_rows = (
            spark.table(
                table_name
            )
            .count()
        )


        record_check(
            "XAI 10",
            f"Table is non-empty: {table_name}",
            actual_rows > 0,
            f"rows={actual_rows}"
        )


# ============================================================
# 22. Final validation table
# ============================================================

print(
    "\n"
    "============================================================"
)
print("18. FINAL VALIDATION SUMMARY")
print(
    "============================================================"
)


validation_schema = [
    "section",
    "check_name",
    "status",
    "details"
]


validation_df = (
    spark.createDataFrame(
        validation_results,
        validation_schema
    )
)


display(
    validation_df
    .orderBy(
        "section",
        "check_name"
    )
)


# ============================================================
# 23. Save final validation results
# ============================================================

VALIDATION_TABLE = (
    "dissertation.lendingclub."
    "lc_2007_2017_final_technical_validation"
)


validation_df.write \
    .mode(
        "overwrite"
    ) \
    .option(
        "overwriteSchema",
        "true"
    ) \
    .format(
        "delta"
    ) \
    .saveAsTable(
        VALIDATION_TABLE
    )


print(
    "\nSaved validation table:",
    VALIDATION_TABLE
)


# ============================================================
# 24. Final technical status
# ============================================================

failure_count = (

    validation_df

    .filter(
        F.col(
            "status"
        )
        ==
        "FAIL"
    )

    .count()
)


pass_count = (

    validation_df

    .filter(
        F.col(
            "status"
        )
        ==
        "PASS"
    )

    .count()
)


print(
    "\n"
    "============================================================"
)

print(
    "FINAL TECHNICAL VALIDATION RESULTS"
)

print(
    "============================================================"
)

print(
    "PASS:",
    pass_count
)

print(
    "FAIL:",
    failure_count
)


if failure_count == 0:

    print(
        "\n"
        "FINAL TECHNICAL VALIDATION: PASSED"
    )

else:

    print(
        "\n"
        "FINAL TECHNICAL VALIDATION: "
        f"{failure_count} CHECK(S) FAILED"
    )