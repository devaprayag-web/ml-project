"""
Preprocessing pipeline.

Design decisions (documented here so you can explain them in interviews):

0. LEAKAGE FIX (v3): every statistic used below - medians, modes, IQR
   fences, which columns count as "constant", which categories a
   OneHotEncoder knows about, which categories are "high cardinality", and
   the target LabelEncoder's class list - is now fit on the TRAINING split
   only and then applied unchanged to the test split. Earlier versions of
   this pipeline ran the full pipeline on the whole dataset *before*
   splitting into train/test, which let information about the test rows
   (their values, their categories, their outliers) leak into numbers that
   are only supposed to come from training data. The effect was usually
   small, but it made the reported test metrics slightly optimistic - not
   the honest, train-only estimate the rest of this codebase (feature
   scaling, KNN's dimensionality reduction) already insists on.

   This is implemented as a `Preprocessor` class with the same fit/transform
   split scikit-learn transformers use: `fit()` looks only at the training
   dataframe and records everything it needs; `transform()` applies those
   frozen values to any dataframe (train or test) without recomputing
   anything from it.

1. Null handling is dtype-aware, and missingness-aware too:
   - Before filling, any feature column with at least one null IN THE
     TRAINING SPLIT gets a companion binary "<col>_was_missing" indicator
     column. Sometimes the fact that a value was missing is itself
     predictive (e.g. a null `income` might correlate with the target in a
     way the median-filled value alone can't capture) - dropping that
     signal by silently imputing was a real information loss in the old
     pipeline.
   - Numeric columns -> filled with the TRAINING MEDIAN (robust to
     outliers, unlike mean)
   - Object/categorical columns -> filled with the TRAINING MODE (most
     frequent value)
   - Columns that are 100% null (in the training split) -> dropped
     entirely (can't impute nothing)
   Fill values are computed for every numeric/categorical column while
   fitting (not just ones with nulls in training), so a column that has no
   nulls in training but does have one in the test split still has a valid
   fill value to fall back on.

2. Outlier handling: numeric FEATURE columns (never the target) are capped
   to the standard IQR fence [Q1 - 1.5*IQR, Q3 + 1.5*IQR], computed from the
   TRAINING split only, rather than having outlier rows dropped. Capping
   (Winsorizing) preserves every row - no risk of accidentally breaking a
   stratified split or losing rare-class rows - while still preventing a
   handful of extreme values from dominating distance calculations (KNN) or
   squared-error loss (Linear/Ridge Regression). The target is deliberately
   left untouched: capping it would silently change what the model is being
   asked to predict.

3. Constant (zero-variance) feature columns - based on the TRAINING split -
   are dropped before encoding - a column with only one distinct value in
   training carries no information for any of the five algorithms and just
   adds dead weight to the feature matrix.

4. Categorical encoding is split by role AND by cardinality:
   - FEATURE columns that are categorical and LOW cardinality (at most
     `HIGH_CARDINALITY_THRESHOLD` distinct values in the training split) ->
     `OneHotEncoder`, fit on training categories only
     (`handle_unknown="ignore"` so an unseen test-time category becomes an
     all-zero row instead of crashing). This avoids the false ordinal
     relationship LabelEncoder would imply (e.g. encoding
     ["Red", "Blue", "Green"] as 0/1/2 wrongly suggests Green > Red).
   - FEATURE columns that are categorical and HIGH cardinality (more than
     `HIGH_CARDINALITY_THRESHOLD` distinct values in training, e.g. a
     `customer_id` column with thousands of unique values) -> FREQUENCY
     (count) ENCODING instead: each category is replaced by how often it
     appeared in the training split (as a proportion of training rows,
     computed once during fit). This turns what would have been thousands
     of new one-hot columns into a single numeric column, at the cost of
     mapping two different categories with the same training frequency to
     the same encoded value. A category seen only at test time gets a
     frequency of 0.0 (it never appeared in training, so a rate of 0 is the
     honest answer). This is a good default for the tree-based algorithms
     (decision tree, random forest) and workable for the others; a target
     encoding with smoothing would generalize better still but needs more
     machinery (out-of-fold fitting) than a v1 needs.
   - The TARGET column, if categorical, still uses LabelEncoder, fit on the
     training split's class list. scikit-learn classifiers require a single
     integer class label per row (0, 1, 2...), not a one-hot vector, so the
     target can never be one-hot or frequency encoded regardless of how
     "correct" that would be for a feature. If the test split contains a
     class the training split never saw (only possible with a very small or
     very imbalanced class), that's surfaced as a clear error rather than a
     silent crash - training on classes the model never saw an example of
     isn't something 5 lines of code should paper over.

5. Feature scaling (StandardScaler) is applied for both Logistic Regression
   and KNN in ml.py, fit on the training split only.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from typing import Tuple, Dict, List, Optional, Any

# A categorical feature column with more distinct training values than this
# skips one-hot encoding (which would otherwise add one column per category)
# and gets frequency-encoded into a single numeric column instead. 20 matches
# the cardinality threshold already used elsewhere in this app (e.g.
# `_looks_continuous` in ml.py) for "this is a lot of distinct values."
HIGH_CARDINALITY_THRESHOLD = 20


def drop_all_null_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop any column that is entirely null. Returns (cleaned_df, dropped_column_names).

    Safe to run before a train/test split: whether a column is 100% null is
    a structural fact about the whole file, not a statistic that could leak
    target-feature relationships across the split.
    """
    all_null_cols = [col for col in df.columns if df[col].isnull().all()]
    return df.drop(columns=all_null_cols), all_null_cols


class Preprocessor:
    """
    Fit/transform preprocessing pipeline, mirroring the scikit-learn
    transformer contract so nothing that depends on the data's distribution
    (medians, modes, IQR fences, one-hot categories, high-cardinality
    frequency maps, target classes) is ever computed using test rows.

    Usage:
        pre = Preprocessor()
        train_processed = pre.fit_transform(df_train, target_column)
        test_processed = pre.transform(df_test)
    """

    def __init__(self, high_cardinality_threshold: int = HIGH_CARDINALITY_THRESHOLD):
        self.high_cardinality_threshold = high_cardinality_threshold
        self.target_column: Optional[str] = None

        # State captured during fit() - nothing here is touched by transform().
        self.dropped_columns: List[str] = []
        self.missingness_indicator_columns: List[str] = []
        self.numeric_fill_values: Dict[str, float] = {}
        self.categorical_fill_values: Dict[str, Any] = {}
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.outlier_capped_counts: Dict[str, int] = {}
        self.low_cardinality_columns: List[str] = []
        self.high_cardinality_columns: List[str] = []
        self.frequency_maps: Dict[str, Dict[str, float]] = {}
        self.ohe: Optional[OneHotEncoder] = None
        self.target_encoder: Optional[LabelEncoder] = None
        self.fitted_columns: Optional[List[str]] = None  # exact output column order from fit
        self._is_fit = False

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        self.target_column = target_column
        df = df.copy()

        # Structural drops - based on the training split only.
        df, null_dropped = drop_all_null_columns(df)
        df, constant_dropped = self._drop_constant_columns(df, target_column)
        self.dropped_columns = null_dropped + constant_dropped

        # Missingness indicators: decide WHICH columns get a flag using
        # training nulls only.
        self.missingness_indicator_columns = [
            f"{col}_was_missing"
            for col in df.columns
            if col != target_column and df[col].isnull().sum() > 0
        ]

        # Fill values: computed for every numeric/categorical column (not
        # just the ones with training nulls), so a column that happens to
        # have a null only in the test split still has somewhere to fall
        # back to.
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_fill_values[col] = float(df[col].median())
            else:
                mode_val = df[col].mode()
                self.categorical_fill_values[col] = mode_val.iloc[0] if not mode_val.empty else "missing"

        df = self._add_missingness_indicators(df)
        df = self._fill_nulls(df)

        # Outlier fences: computed from training feature columns only.
        for col in df.columns:
            if col == target_column or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if col in self.missingness_indicator_columns:
                continue  # binary flag column, capping is meaningless
            if df[col].nunique(dropna=True) <= 2:
                continue  # already-binary column
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            n_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_capped > 0:
                self.outlier_bounds[col] = (lower, upper)
                self.outlier_capped_counts[col] = n_capped

        df = self._apply_outlier_bounds(df)

        # Categorical encoding: split feature columns by cardinality
        # (measured on the training split), fit target LabelEncoder on
        # training classes.
        if (
            target_column is not None
            and target_column in df.columns
            and not pd.api.types.is_numeric_dtype(df[target_column])
        ):
            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(df[target_column].astype(str))

        categorical_features = [
            col for col in df.columns
            if col != target_column and not pd.api.types.is_numeric_dtype(df[col])
        ]
        for col in categorical_features:
            if df[col].nunique(dropna=True) > self.high_cardinality_threshold:
                self.high_cardinality_columns.append(col)
            else:
                self.low_cardinality_columns.append(col)

        if self.high_cardinality_columns:
            for col in self.high_cardinality_columns:
                self.frequency_maps[col] = (
                    df[col].astype(str).value_counts(normalize=True).to_dict()
                )

        if self.low_cardinality_columns:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.ohe.fit(df[self.low_cardinality_columns].astype(str))

        self._is_fit = True
        transformed = self._transform_encoded(df)
        self.fitted_columns = transformed.columns.tolist()
        return transformed

    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        return self.fit(df, target_column)

    # ------------------------------------------------------------------ #
    # transform
    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError("Preprocessor.transform() called before fit().")

        df = df.copy()
        # Drop exactly the columns fit() decided to drop - never recompute
        # "is this column all-null / constant?" from the split being
        # transformed.
        df = df.drop(columns=[c for c in self.dropped_columns if c in df.columns])

        df = self._add_missingness_indicators(df)
        df = self._fill_nulls(df)
        df = self._apply_outlier_bounds(df)

        transformed = self._transform_encoded(df)

        # Guarantee identical columns/order to the fitted training output -
        # covers the rare case where a categorical column's OneHotEncoder
        # output could differ (it won't, thanks to handle_unknown="ignore",
        # but this keeps X_train/X_test alignment airtight regardless).
        if self.fitted_columns is not None:
            for col in self.fitted_columns:
                if col not in transformed.columns:
                    transformed[col] = 0
            transformed = transformed[self.fitted_columns]
        return transformed

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _drop_constant_columns(
        df: pd.DataFrame, target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Drop feature columns with exactly one distinct (non-null) value in
        this (training) split. The target is never dropped this way, even if
        constant - that's a data-quality problem worth a training-time error,
        not a silent drop."""
        constant_cols = [
            col for col in df.columns
            if col != target_column and df[col].nunique(dropna=True) <= 1
        ]
        return df.drop(columns=constant_cols), constant_cols

    def _add_missingness_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for indicator_name in self.missingness_indicator_columns:
            col = indicator_name[: -len("_was_missing")]
            if col in df.columns:
                df[indicator_name] = df[col].isnull().astype(int)
            else:
                df[indicator_name] = 0
        return df

    def _fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if col in self.numeric_fill_values:
                df[col] = df[col].fillna(self.numeric_fill_values[col])
            elif col in self.categorical_fill_values:
                df[col] = df[col].fillna(self.categorical_fill_values[col])
            elif df[col].isnull().any():
                # A column that appeared only after fit (shouldn't normally
                # happen) - fall back to a dtype-aware default rather than
                # leaving nulls in place.
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna("missing")
        return df

    def _apply_outlier_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def _transform_encoded(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply target LabelEncoder + one-hot/frequency feature encoding
        using ONLY state captured during fit()."""
        df = df.copy()

        if (
            self.target_encoder is not None
            and self.target_column is not None
            and self.target_column in df.columns
        ):
            target_values = df[self.target_column].astype(str)
            unseen = set(target_values.unique()) - set(self.target_encoder.classes_)
            if unseen:
                raise ValueError(
                    f"Target column '{self.target_column}' contains class(es) {sorted(unseen)} "
                    "that never appeared in the training split, so the model never learned "
                    "them. This usually means a class has too few rows to be represented in "
                    "both the train and test split - add more examples of that class, or "
                    "remove it from the dataset."
                )
            df[self.target_column] = self.target_encoder.transform(target_values)

        if self.high_cardinality_columns:
            for col in self.high_cardinality_columns:
                freq_map = self.frequency_maps[col]
                encoded_name = f"{col}_freq_encoded"
                df[encoded_name] = df[col].astype(str).map(freq_map).fillna(0.0)
            df = df.drop(columns=[c for c in self.high_cardinality_columns if c in df.columns])

        if self.low_cardinality_columns and self.ohe is not None:
            present_cols = [c for c in self.low_cardinality_columns if c in df.columns]
            encoded_array = self.ohe.transform(df[present_cols].astype(str))
            encoded_columns = self.ohe.get_feature_names_out(present_cols)
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)
            df = df.drop(columns=present_cols)
            df = pd.concat([df, encoded_df], axis=1)

        return df

    # ------------------------------------------------------------------ #
    # reporting helpers - used by ml.py to populate the API's `extras`
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        return {
            "dropped_columns": self.dropped_columns,
            "missingness_indicators_added": self.missingness_indicator_columns,
            "outliers_capped": dict(self.outlier_capped_counts),
            "high_cardinality_columns_frequency_encoded": self.high_cardinality_columns,
            "high_cardinality_threshold": self.high_cardinality_threshold,
        }


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit scaler on train, apply to both train and test (never fit on test - avoids data leakage)."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler
