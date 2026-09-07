## 4. Train/Test Split

Method:

`randomSplit([0.8, 0.2], seed=42)`

Final counts:

- Training observations: 364,399
- Test observations: 90,878

Training target distribution:

- Non-default: 271,018
- Default: 93,381

Test target distribution:

- Non-default: 67,393
- Default: 23,485

The complete dataset contained 338,411 non-default observations
(74.33%) and 116,866 default observations (25.67%).


## 5. Final Models

### Logistic Regression

- `maxIter = 50`
- `regParam = 0.0`
- `elasticNetParam = 0.0`

### Decision Tree

- `maxDepth = 8`
- `maxBins = 32`
- `minInstancesPerNode = 1`

### Random Forest

- `numTrees = 50`
- `maxDepth = 12`
- `maxBins = 32`
- `featureSubsetStrategy = auto`

### Multilayer Perceptron

- Architecture: `[135, 64, 32, 2]`
- `maxIter = 50`
- `blockSize = 256`
- `solver = l-bfgs`


## 9. DTI Validation

Extreme DTI observations were investigated as part of the final
data-quality validation.

The final dataset contained:

- 455,277 total observations
- 232 observations with DTI >= 60
- 75 observations with DTI >= 100
- 5 observations with DTI = 999

The median DTI was 17.05, the 99th percentile was 37.25,
and the 99.9th percentile was 48.98.

All five observations with DTI = 999 were compared with the raw
LendingClub source table. In every case, the raw value was also
999. Therefore, these extreme values were present in the source
data and were not introduced by cleaning, casting, missing-value
imputation or feature engineering.

Additional extreme values including 886.77, 818.10, 641.36,
595.24 and 532 were also present, indicating a sparse extreme
upper tail rather than a preprocessing error isolated to the
value 999.

All five DTI=999 observations occurred among joint applications
from the 2016–2017 source period. Because there was insufficient
evidence to classify 999 as an invalid or missing-value code,
the observations were retained.

This issue is documented as a source-data limitation and is
considered when interpreting extreme local SHAP cases.
