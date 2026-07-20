catalog_name = "dissertation"
schema_name = "lendingclub"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {schema_name}")

spark.sql("SHOW CATALOGS").show(truncate=False)
spark.sql("SHOW SCHEMAS IN dissertation").show(truncate=False)
spark.sql("SELECT current_catalog(), current_schema()").show(truncate=False)