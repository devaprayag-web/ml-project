# No-Code ML Trainer — Backend

A FastAPI service that takes an uploaded CSV, lets you pick a target column
and an algorithm (Linear Regression, Logistic Regression, KNN, or Decision
Tree), and returns trained model metrics + chart-ready data.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000`. Interactive API docs (auto-generated
by FastAPI) at `http://127.0.0.1:8000/docs` — use this to test endpoints
manually before wiring up the frontend.

## API

### `POST /api/upload`
Multipart form upload, field name `file`, must be `.csv`.

Returns: `session_id`, row count, per-column info (name, dtype, null count,
unique count), a 5-row preview, and any columns dropped for being 100% null.

### `POST /api/train`
```json
{
  "session_id": "uuid-from-upload",
  "algorithm": "linear_regression | logistic_regression | knn | decision_tree",
  "target_column": "column_name",
  "k": 5
}
```
`k` is optional, only used by KNN (defaults to 5).

Returns: `problem_type`, `metrics`, `extras` (algorithm-specific data — see
below), and `sample_predictions` (10 test-set rows: actual vs predicted).

## Design decisions (read this before extending the project)

**Problem type mapping**: Linear Regression → always regression. Logistic
Regression → always classification. KNN and Decision Tree → **auto-detected**
from the target column: a continuous numeric target (a float column, or an
integer column with more than 20 distinct values) routes to
`KNeighborsRegressor`/`DecisionTreeRegressor`; anything else (text, or a
small set of discrete integers) routes to the classifier variant. This
reuses the exact same "does this look continuous?" heuristic already used
to block Logistic Regression on a continuous target — see `get_problem_type()`
in `app/ml.py`.

One tradeoff worth naming: a numeric column with many discrete categories
(e.g. a "zip code" stored as an int with 30+ unique values) would be
auto-detected as regression even though it's really a category. This is the
same ambiguity Logistic Regression's guard already can't resolve perfectly —
there's no way to distinguish "continuous measurement" from "discrete code
with a lot of values" from dtype and cardinality alone.

**Null handling**: dtype-aware. Numeric columns filled with **median**
(robust to outliers, unlike mean). Categorical columns filled with **mode**.
Columns that are 100% null are dropped entirely — you can't impute nothing.

**Categorical encoding**: split by role, and (as of Round 3, see below) by
cardinality.
- **Feature columns** that are categorical and low-cardinality (≤20 distinct
  values in the training split) → `OneHotEncoder` (one binary column per
  category). Avoids the false ordinal relationship a single integer column
  would imply for non-ordinal categories like `["Red", "Blue", "Green"]`
  (encoding them 0/1/2 would wrongly suggest Green > Red).
- **Feature columns** that are categorical and high-cardinality (>20
  distinct values, e.g. a `customer_id` column with 10,000 unique values) →
  frequency (count) encoding instead — see Round 3 below for why and how.
- **Target column**, if categorical → `LabelEncoder` (single integer class
  per row), because every classifier here (Logistic Regression, KNN,
  Decision Tree, Random Forest) requires one integer label to predict, not
  a one-hot vector. This also keeps a fitted encoder around to decode
  predictions back into readable class names for the confusion matrix and
  sample predictions.

**Feature scaling**: `StandardScaler` is applied only for KNN, fit on the
training split only (never on test — avoids data leakage), since KNN is
distance-based and unscaled features on wildly different ranges (age vs.
salary) would dominate the distance calculation.

**Chart data selection**:
- Linear Regression with exactly 1 feature column → `line_scatter` (x,
  actual, predicted) — the classic line-through-points plot.
- Linear Regression with 2+ feature columns → `actual_vs_predicted` scatter
  (generalizes to any number of features; a perfect model's points sit on
  the diagonal).
- Logistic Regression → confusion matrix (always) + coefficients per
  feature (interpretability).
- Decision Tree → confusion matrix (always) + feature importances.
- KNN → confusion matrix (always). If exactly 2 feature columns remain
  after preprocessing, also returns a `decision_boundary_grid` (30x30 mesh
  of predictions) for the classic KNN region-coloring visual.

**Sessions**: in-memory dict, keyed by UUID, no database. Fine for a
portfolio project as a single process. Resets on server restart — swap for
Redis if you ever need it to survive restarts or scale to multiple workers.

## Bugs found and fixed during real-world testing

Two edge cases surfaced when testing against real datasets (not just synthetic
test data) — worth knowing about since they're the kind of thing that looks
obvious in hindsight but isn't caught by a quick smoke test:

1. **Classifiers on continuous numeric targets.** Picking Logistic Regression,
   KNN, or Decision Tree with a continuous numeric target (e.g. a house
   `price` column) used to surface scikit-learn's raw internal error
   ("Unknown label type: continuous") instead of a clear message. Fixed by
   checking the target's dtype and cardinality *before* training — a float
   column, or an integer column with more than ~20 distinct values, is
   treated as continuous and blocked for classification algorithms with a
   plain-English explanation.

2. **All-same-dtype datasets crashing on upload.** A CSV where every column
   happens to be `int64` (no float or text columns at all) used to crash
   `/api/upload` with a `PydanticSerializationError`. Root cause: pandas'
   `iterrows()` returns raw `numpy.int64` scalars when a DataFrame's columns
   are all one dtype, but coerces to native Python types when columns are
   mixed dtypes — an inconsistency that only shows up on homogeneous-dtype
   data. Fixed by explicitly calling `.item()` on any value that has it
   before returning it as JSON, regardless of dtype mix.

## Model tuning (accuracy fixes, all via cross-validation on the training set only)

Every algorithm previously ran with sklearn defaults / a single hand-picked
parameter. Each now gets its hyperparameters chosen automatically via
cross-validation — never touching the held-out test set during the search,
so the reported test metric stays an honest estimate of generalization.

- **Linear Regression → RidgeCV.** Plain OLS was unstable when one-hot
  encoding produced correlated dummy columns (multicollinearity). RidgeCV
  picks the regularization strength (`alpha`) automatically from a wide
  log-spaced range; if the data doesn't need regularization it lands on a
  tiny alpha and behaves like OLS anyway — strictly no worse than before.
- **Logistic Regression** now (1) gets `StandardScaler`-scaled features —
  previously only KNN was scaled, which biased L2 regularization unevenly
  across features on different scales — and (2) searches `C` (regularization
  strength) and `class_weight` (`None` vs `'balanced'`) via `GridSearchCV`
  instead of sitting at sklearn's default `C=1.0`.
- **Decision Tree** was running fully unconstrained (`random_state=42` only),
  which grows the tree until every leaf is pure — memorizing the training
  set. Now searches `max_depth`, `min_samples_split`, `min_samples_leaf`,
  `criterion`, and (for classification) `class_weight` via `GridSearchCV`.
  Verified on a synthetic test set: the old unconstrained tree hit 100%
  train accuracy but only 77.5% test accuracy (classic overfitting); the
  tuned tree (`max_depth=3`) scored 80% on the same test split.
- **KNN** previously only tuned `k` (via a hand-rolled elbow-method CV
  search). `weights` (`'uniform'` vs `'distance'`) and `p` (1=Manhattan,
  2=Euclidean) were left at sklearn defaults. Now all three are searched
  jointly via `GridSearchCV`. The `/api/train` elbow-chart response
  (`k_search`, `k_search_metric`) is unchanged in shape — it's regenerated
  at the winning `weights`/`p` after the joint search, so the existing
  frontend chart keeps working without changes.
- Every algorithm's `extras` now includes `tuned_params` (or
  `regularization_alpha` for Linear Regression) so the chosen
  hyperparameters are visible in the API response and shown in the UI.
- **Train/test split is now stratified for classification** (`stratify=y`),
  falling back to a plain split only if a class has too few members to
  stratify. Previously a plain random split could leave train/test class
  proportions mismatched, adding noise to both training and the reported
  accuracy.

None of this guarantees a specific accuracy number — a dataset with weak
signal in its features is still a dataset with weak signal — but every fix
targets a real, previously-present source of avoidable error (unscaled
regularization, unconstrained tree overfitting, undertuned KNN, an
imbalance-blind split).

## Round 2: preprocessing, evaluation, and a 5th algorithm

- **Missingness indicators.** Before imputing, any feature column with at
  least one null gets a companion `<col>_was_missing` binary column
  (`preprocessing.py::add_missingness_indicators`). Median/mode filling
  alone throws away the fact that a value was ever missing - sometimes
  that fact is itself predictive.
- **Outlier capping (Winsorizing).** Numeric FEATURE columns are clamped to
  the standard IQR fence `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` before training
  (`preprocessing.py::cap_outliers_iqr`) - rows are never dropped, only
  extreme values are clamped, so stratified splits and small classes are
  never put at risk. **The target column is deliberately excluded** -
  capping it would silently change what the model is being asked to
  predict. This means a regression target with its own extreme outliers is
  still not handled (see limitations below).
- **Constant (zero-variance) columns are dropped** before encoding - they
  carry no information for any algorithm.
- **Final k-fold CV summary.** Every algorithm now also reports a
  mean/std accuracy or R² from cross-validating the winning hyperparameters
  on the training set (`extras.cv_validation`), alongside the single
  80/20-split test metric - a more trustworthy read on generalization than
  one fixed split alone.
- **ROC-AUC and PR-AUC** are computed for every classifier that exposes
  `predict_proba` (all of them), with a full ROC curve returned for binary
  targets. More informative than accuracy alone under class imbalance -
  exactly what `class_weight='balanced'` targets.
- **Linear Regression now compares RidgeCV against ElasticNetCV** via
  nested cross-validation and keeps whichever generalizes better.
  ElasticNet can zero out irrelevant one-hot columns entirely (true
  feature selection); Ridge only shrinks them.
- **Logistic Regression now searches `penalty` (L1 vs L2) too**, via
  `solver='saga'` (the only sklearn solver supporting both in one
  estimator), alongside `C` and `class_weight`.
- **Decision Tree's grid now scales with dataset size** (`max_depth`
  options derived from `log2(n_samples)` instead of fixed guesses) and adds
  `ccp_alpha` (cost-complexity pruning) as an alternative/complementary way
  to control overfitting.
- **Bounded search (`_bounded_search`)**: grids small enough to search
  exhaustively still use `GridSearchCV`; larger grids (Random Forest,
  wider Decision Tree grids) automatically switch to `RandomizedSearchCV`
  with a capped `n_iter`, so training time stays predictable regardless of
  how large a parameter grid gets.
- **KNN's feature set is capped at 20 dimensions** via `SelectKBest`
  (mutual information) fit on the training set only, when preprocessing
  produces more than that - KNN suffers the most of the five algorithms
  from high-dimensional one-hot spaces (curse of dimensionality). Also:
  feature scaling is now **skipped** for KNN when every feature is already
  binary (a fully one-hot feature set) - scaling would rescale each column
  by how common its category is, which distorts Manhattan distance's
  natural equivalence to Hamming distance on pure 0/1 data. (A separate
  `metric='hamming'` search branch was considered and deliberately **not**
  added: for 0/1 feature vectors, Manhattan distance IS Hamming distance up
  to a constant scale factor, so it produces identical nearest-neighbor
  rankings - the existing `p=1` search already covers this case.)
- **Random Forest added as a 5th algorithm** (`app/ml.py::_tune_random_forest`,
  `Algorithm.random_forest` in `schemas.py`). Averaging many decorrelated
  trees typically beats even a well-tuned single Decision Tree on test
  accuracy, at the cost of the single tree's easy visual interpretability.
  Tuned via the bounded search above (`n_estimators` capped at 200,
  `n_iter=20`) specifically because an exhaustive grid measured 35-40
  seconds per training request on a single-core test environment - too
  slow for what should feel like a fast, interactive wizard. Expect Random
  Forest training to still take several seconds (it's the most expensive
  of the five); the other four remain sub-2-seconds typically.

## Round 3: leakage-free preprocessing, high-cardinality encoding, model export

- **Preprocessing no longer sees the test split.** Previously
  `preprocess_dataframe()` ran on the *entire* uploaded dataset before the
  80/20 train/test split happened - so the median/mode fill values, IQR
  outlier fences, constant-column detection, and one-hot category discovery
  were all computed using rows that later ended up in the test set. That's
  a classic (if subtle) leakage bug: it made the reported test metrics
  slightly optimistic, contradicting the "never touches test data" guarantee
  the rest of the pipeline (feature scaling, KNN's dimensionality reduction)
  already followed. Fixed by rewriting preprocessing as a `Preprocessor`
  class (`app/preprocessing.py`) with the same fit/transform contract
  scikit-learn transformers use: `train_and_evaluate()` now splits the raw
  dataframe *first*, calls `Preprocessor.fit()` on the training split only,
  and calls `Preprocessor.transform()` on the test split - every statistic
  (medians, modes, IQR bounds, one-hot categories, frequency maps, the
  target's `LabelEncoder` classes) is frozen from training data and applied
  mechanically to test rows, including test-only nulls and unseen
  categories.
- **High-cardinality categorical columns no longer blow up one-hot width.**
  A categorical feature column is now one-hot encoded only if it has at
  most `HIGH_CARDINALITY_THRESHOLD` (20) distinct values in the training
  split. Above that - e.g. a `customer_id` column with thousands of unique
  values - it's **frequency-encoded** instead: each category becomes a
  single numeric column equal to how often that category appeared in
  training (as a proportion), computed once during `fit()`. A category seen
  only at test time gets a frequency of 0.0. This turns what would have
  been thousands of new columns into one, at the cost of two different
  categories with the same training frequency mapping to the same encoded
  value - a fine trade for the tree-based algorithms in particular. The
  columns affected are reported in the API response as
  `extras.high_cardinality_columns_frequency_encoded`.
- **Model export via joblib.** Every successful `/api/train` call now also
  fits an exportable bundle - the trained estimator, the fitted
  `Preprocessor`, the fitted `StandardScaler` (if the algorithm used one),
  and the fitted KNN feature selector (if dimensionality reduction ran) -
  stored server-side (in-memory, see `app/model_store.py`) under a fresh
  `model_id` returned in the training response. `GET
  /api/export/{model_id}` streams that bundle back as a `.joblib` file.
  Loading it elsewhere (`joblib.load(...)`) gives back a dict with
  `preprocessor`, `scaler`, `knn_selector`, and `model` keys, applied in
  that order to preprocess and predict on brand-new raw rows the exact same
  way the training pipeline did.

## Known limitations / good v2 features to mention in an interview

- **Target-column outliers are not handled.** Outlier capping is
  deliberately scoped to feature columns only (see above) - a regression
  target with its own extreme values will still pull the fit toward those
  values. Handling this well needs a decision (drop those rows? cap the
  target and document the change in problem definition? use a
  robust/quantile loss?) rather than a silent default.
- **Frequency encoding is a v1, not the ceiling.** It's a one-line-per-row
  transform with no leakage risk, but it can't distinguish two categories
  that happen to occur equally often. A target-mean encoding with smoothing
  (fit via out-of-fold CV on the training set) would generalize better for
  high-cardinality columns that actually correlate with the target - a good
  v2 addition once the extra fitting machinery is worth it.
- No persistence — sessions AND exported models are lost on server restart
  (`model_store.py` has the identical in-memory-dict tradeoff as
  `session_store.py` - swap both for Redis/a real store to survive
  restarts or scale to multiple workers)
- Auto-detecting regression vs classification for KNN/Decision
  Tree/Random Forest from target cardinality can misfire on a numeric
  "code" column with many distinct values (e.g. zip codes) — same
  ambiguity noted above
- No support for `.xlsx` uploads yet, CSV only
- `GridSearchCV`/`RandomizedSearchCV` now run on every training request,
  so training takes seconds rather than being instant - acceptable for a
  wizard tool, but would need caching or a background job queue for a
  production system training on large datasets repeatedly
- The exported `.joblib` bundle contains a live scikit-learn estimator -
  only unpickle/joblib.load bundles you trust (this is a general
  joblib/pickle caveat, not something specific to this app)

