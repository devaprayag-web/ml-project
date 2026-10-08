"""
Training + evaluation logic for all 5 algorithms.

Problem type mapping:
  - linear_regression      -> always regression
  - logistic_regression    -> always classification
  - knn, decision_tree,
    random_forest          -> FLEXIBLE: auto-detected from the target column.
    A continuous numeric target (a float column, or an integer column with
    more than 20 distinct values) -> regression. Anything else (text, or a
    small set of discrete integers) -> classification.

TUNING (all five algorithms are hyperparameter-tuned via cross-validation on
the training set only - never on the held-out test set):
  - linear_regression   -> RidgeCV vs ElasticNetCV, chosen via nested CV
  - logistic_regression -> scaled features + GridSearchCV over C/penalty/class_weight
  - decision_tree       -> bounded search over depth/split/leaf/criterion/ccp_alpha
  - random_forest       -> bounded search over n_estimators/depth/split/leaf/max_features
  - knn                 -> GridSearchCV over n_neighbors/weights/p + dimensionality
                            reduction for high-dimensional feature sets

Every algorithm also gets a final k-fold CV summary (mean/std) computed on
the training set with its winning hyperparameters, reported alongside the
single-split test metrics - a more trustworthy signal of how well the
chosen model actually generalizes than one fixed split alone.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import RidgeCV, ElasticNetCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve,
)
from typing import Dict, Any, Tuple, Optional

from .preprocessing import Preprocessor, scale_features

TREE_LIKE_ALGORITHMS = ("decision_tree", "random_forest")


def _looks_continuous(col: pd.Series) -> bool:
    """A float column, or an integer column with many distinct values, is
    almost certainly a continuous measurement rather than a category."""
    return pd.api.types.is_float_dtype(col) or col.nunique() > 20


def get_problem_type(algorithm: str, target_col: Optional[pd.Series] = None) -> str:
    """
    Determine regression vs classification for a given algorithm + target.

    linear_regression and logistic_regression are fixed. knn, decision_tree,
    and random_forest are flexible and require target_col to infer the
    right problem type from the target column's dtype/cardinality.
    """
    if algorithm == "linear_regression":
        return "regression"
    if algorithm == "logistic_regression":
        return "classification"

    if target_col is None:
        raise ValueError(f"target_col is required to infer problem type for '{algorithm}'.")
    if not pd.api.types.is_numeric_dtype(target_col):
        return "classification"
    return "regression" if _looks_continuous(target_col) else "classification"


def _safe_cv_folds(y_train: pd.Series, problem_type: str, max_folds: int = 5) -> int:
    """
    Shared by every tuning function below. Classification uses
    StratifiedKFold internally, which requires every fold to contain at
    least one sample of every class - so cv can't exceed the smallest
    class's sample count. Regression has no such constraint, just a sane
    cap/floor.
    """
    n_samples = len(y_train)
    if problem_type == "classification":
        smallest_class_count = int(pd.Series(y_train).value_counts().min())
        return max(2, min(max_folds, smallest_class_count, n_samples))
    return max(2, min(max_folds, n_samples))


def _bounded_search(estimator, param_grid: Dict[str, list], cv: int, scoring: str,
                     max_iter: int = 40, n_jobs: int = -1, random_state: int = 42):
    """
    Exhaustive GridSearchCV for small grids; RandomizedSearchCV once the
    grid is large enough that an exhaustive search would make every
    training request noticeably slower. Because param_grid values are
    plain lists, RandomizedSearchCV samples WITHOUT replacement - if the
    grid is smaller than max_iter it just runs every combination anyway
    (with a harmless warning), so this never actually skips an exhaustive
    search when the grid is already small.
    """
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    if total_combinations <= max_iter:
        return GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    return RandomizedSearchCV(
        estimator, param_grid, n_iter=max_iter, cv=cv, scoring=scoring,
        n_jobs=n_jobs, random_state=random_state,
    )


def train_and_evaluate(
    df: pd.DataFrame, algorithm: str, target_column: str, k: int = 5, auto_k: bool = False
) -> Tuple[str, Dict[str, Any], Dict[str, Any], list, Dict[str, Any]]:
    """
    Runs the full pipeline: detect problem type -> split -> fit preprocessing
    on the TRAINING split only -> (scale/reduce if needed) -> tune + train ->
    evaluate.

    LEAKAGE FIX: the train/test split now happens on the raw dataframe,
    before any preprocessing. Preprocessor.fit() only ever sees the training
    rows - medians, modes, IQR fences, one-hot categories, high-cardinality
    frequency maps, and the target's LabelEncoder classes are computed
    exclusively from training data, then applied unchanged to the test split
    via Preprocessor.transform(). Previously the entire pipeline ran on the
    full dataset before splitting, which let statistics computed from
    would-be test rows leak into the numbers used to train and evaluate.

    Returns (problem_type, metrics, extras, sample_predictions, model_artifact).
    model_artifact bundles everything needed to reuse the trained model later
    (see app/model_store.py / the /api/export endpoint): the fitted
    estimator, the fitted Preprocessor (so raw new rows can be transformed
    the same way), and bookkeeping fields (algorithm, target_column,
    feature_names_used, problem_type).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    original_target_is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
    problem_type = get_problem_type(algorithm, df[target_column])

    if problem_type == "regression" and not original_target_is_numeric:
        raise ValueError(
            f"Linear Regression requires a numeric target column. "
            f"'{target_column}' contains text/categorical values - "
            "pick a classification algorithm instead, or choose a numeric target."
        )

    if algorithm == "logistic_regression" and original_target_is_numeric and _looks_continuous(df[target_column]):
        raise ValueError(
            f"'{target_column}' looks like a continuous numeric column "
            f"({df[target_column].nunique()} distinct values), not a set of categories. "
            "Logistic Regression needs a target with a small number of discrete classes. "
            "Use Linear Regression instead, or choose a categorical/discrete target column "
            "(KNN, Decision Tree, and Random Forest handle continuous targets automatically)."
        )

    # Split the RAW dataframe first - everything preprocessing does from
    # here on is fit on df_train only. Stratify on the raw (unencoded)
    # target for classification; falls back to a plain split if any class
    # has too few members to stratify.
    stratify_arg = df[target_column] if problem_type == "classification" else None
    try:
        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=stratify_arg
        )
    except ValueError:
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    preprocessor = Preprocessor()
    processed_train = preprocessor.fit_transform(df_train, target_column)

    if target_column in preprocessor.dropped_columns:
        raise ValueError(
            f"Target column '{target_column}' was entirely null and had to be dropped. "
            "Choose a different target column."
        )

    processed_test = preprocessor.transform(df_test)
    target_encoder = preprocessor.target_encoder

    X_train = processed_train.drop(columns=[target_column])
    y_train = processed_train[target_column]
    X_test = processed_test.drop(columns=[target_column])
    y_test = processed_test[target_column]

    if X_train.shape[1] == 0:
        raise ValueError("No feature columns remain after preprocessing.")

    feature_names_used = X_train.columns.tolist()

    preprocessing_summary = preprocessor.summary()
    extras: Dict[str, Any] = {
        "dropped_columns": preprocessing_summary["dropped_columns"],
        "missingness_indicators_added": preprocessing_summary["missingness_indicators_added"],
        "outliers_capped": preprocessing_summary["outliers_capped"],
        "high_cardinality_columns_frequency_encoded": preprocessing_summary[
            "high_cardinality_columns_frequency_encoded"
        ],
        "high_cardinality_threshold": preprocessing_summary["high_cardinality_threshold"],
    }

    scaler_used = None
    knn_selector = None

    if algorithm == "linear_regression":
        model, tuning_extras = _tune_linear_regression(X_train, y_train)
    elif algorithm == "logistic_regression":
        # Previously ONLY knn got scale_features() - a real gap, since L2
        # regularization penalizes unscaled features unevenly.
        X_train, X_test, scaler_used = scale_features(X_train, X_test)
        model, tuning_extras = _tune_logistic_regression(X_train, y_train)
    elif algorithm == "knn":
        is_all_binary = bool(X_train.apply(lambda c: set(c.dropna().unique()) <= {0, 1}).all())
        if is_all_binary:
            # Every feature is already 0/1 (a fully one-hot-encoded feature
            # set). Skip StandardScaler here: scaling would rescale each
            # column by how common its category is, which distorts Manhattan
            # distance's natural equivalence to Hamming distance on pure
            # binary data. (A separate 'hamming' metric branch would be
            # redundant with the existing p=1/Manhattan search below, since
            # for 0/1 vectors Manhattan distance IS Hamming distance up to a
            # constant scale factor - same nearest-neighbor ranking either way.)
            tuning_extras_scaling = {"scaling_skipped_reason": "all_features_binary"}
        else:
            X_train, X_test, scaler_used = scale_features(X_train, X_test)
            tuning_extras_scaling = {}
        X_train, X_test, feature_names_used, reduction_extras, knn_selector = _reduce_dimensionality_for_knn(
            X_train, X_test, y_train, problem_type
        )
        model, tuning_extras = _tune_knn(X_train, y_train, problem_type, k, auto_k)
        tuning_extras.update(reduction_extras)
        tuning_extras.update(tuning_extras_scaling)
    elif algorithm == "decision_tree":
        model, tuning_extras = _tune_decision_tree(X_train, y_train, problem_type)
    elif algorithm == "random_forest":
        model, tuning_extras = _tune_random_forest(X_train, y_train, problem_type)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    extras.update(tuning_extras)
    extras.update(_final_cv_summary(model, X_train, y_train, problem_type))

    y_pred = model.predict(X_test)

    if problem_type == "regression":
        metrics = _regression_metrics(y_test, y_pred)
        extras.update(_regression_extras(X_test, y_test, y_pred))
        extras.update(_regression_algorithm_extras(algorithm, model, feature_names_used))
    else:
        metrics = _classification_metrics(y_test, y_pred, target_encoder)
        proba_metrics, proba_extras = _probabilistic_classification_info(model, X_test, y_test)
        metrics.update(proba_metrics)
        extras.update(proba_extras)
        extras.update(_classification_extras(algorithm, model, feature_names_used, X_test, target_encoder))

    # Bundled for the /api/export endpoint (see app/model_store.py): everything
    # needed to preprocess a fresh raw row the same way and get a prediction
    # back out, without re-running any training-time fitting.
    model_artifact: Dict[str, Any] = {
        "model": model,
        "preprocessor": preprocessor,
        "scaler": scaler_used,
        "knn_selector": knn_selector,
        "algorithm": algorithm,
        "problem_type": problem_type,
        "target_column": target_column,
        "feature_names_used": feature_names_used,
    }

    sample_predictions = _sample_predictions(
        X_test, y_test, y_pred, target_column, target_encoder, problem_type
    )

    return problem_type, metrics, extras, sample_predictions, model_artifact


def _final_cv_summary(model, X_train: pd.DataFrame, y_train: pd.Series, problem_type: str) -> Dict[str, Any]:
    """
    The single 80/20 test split reported elsewhere has variance - a
    different random split could give a noticeably different number by
    chance. This re-runs the WINNING hyperparameters (via clone(), so it's
    an unfitted copy) through k-fold cross-validation on the training set
    and reports mean +/- std, giving a more trustworthy sense of how the
    chosen model actually generalizes than the single-split number alone.
    """
    cv_folds = _safe_cv_folds(y_train, problem_type)
    scoring = "r2" if problem_type == "regression" else "accuracy"
    metric_label = "r2_score" if problem_type == "regression" else "accuracy"
    try:
        scores = cross_val_score(clone(model), X_train, y_train, cv=cv_folds, scoring=scoring)
    except ValueError:
        return {}
    return {
        "cv_validation": {
            "metric": metric_label,
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "folds": cv_folds,
        }
    }


def _tune_linear_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Tries both RidgeCV (L2) and ElasticNetCV (L1+L2 blend) and keeps
    whichever generalizes better, measured via NESTED cross-validation: each
    candidate is itself an unfitted CV-tuning estimator, scored across outer
    folds, so the alpha/l1_ratio selection happens fresh inside every outer
    fold rather than leaking into the comparison. ElasticNet can zero out
    coefficients entirely (true feature selection) - useful when
    OneHotEncoder produced some genuinely irrelevant dummy columns; Ridge
    only shrinks toward zero. If plain regularization is all the data needs,
    Ridge usually wins this comparison; if some features are dead weight,
    ElasticNet usually wins.
    """
    n_samples = len(X_train)
    inner_cv = min(5, n_samples) if n_samples >= 3 else None
    outer_cv = _safe_cv_folds(y_train, "regression")
    alphas = np.logspace(-3, 3, 13)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

    def _ridge():
        return RidgeCV(alphas=alphas, cv=inner_cv)

    def _elastic():
        return ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=inner_cv, max_iter=5000, random_state=42)

    use_elastic = False
    try:
        ridge_scores = cross_val_score(_ridge(), X_train, y_train, cv=outer_cv, scoring="r2")
        elastic_scores = cross_val_score(_elastic(), X_train, y_train, cv=outer_cv, scoring="r2")
        use_elastic = elastic_scores.mean() > ridge_scores.mean()
    except ValueError:
        pass  # too few samples to nest CV - fall back to Ridge below

    if use_elastic:
        model = _elastic()
        model.fit(X_train, y_train)
        return model, {
            "regression_method": "elastic_net",
            "regularization_alpha": round(float(model.alpha_), 6),
            "l1_ratio": round(float(model.l1_ratio_), 4),
        }
    else:
        model = _ridge()
        model.fit(X_train, y_train)
        return model, {
            "regression_method": "ridge",
            "regularization_alpha": round(float(model.alpha_), 6),
        }


def _tune_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Searches C (regularization strength), penalty (L1 vs L2), and
    class_weight jointly. Fixed to solver='saga' since it's the only
    scikit-learn solver that supports both L1 and L2 penalties in one
    estimator (avoids having to branch solver choice by penalty). L1 can
    zero out irrelevant one-hot columns entirely; L2 only shrinks them.
    max_iter raised to 3000 since saga can need more iterations than lbfgs
    to converge, especially early in the search before scaling's benefit
    is combined with the right C.
    """
    cv_folds = _safe_cv_folds(y_train, "classification")
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "class_weight": [None, "balanced"],
    }
    search = GridSearchCV(
        LogisticRegression(max_iter=3000, solver="saga"),
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, {"tuned_params": search.best_params_}


def _decision_tree_param_grid(n_samples: int, problem_type: str) -> Dict[str, list]:
    """
    max_depth options scale with dataset size instead of being fixed guesses
    - a 50-row dataset and a 50,000-row dataset shouldn't search the same
    depth range. ccp_alpha (cost-complexity pruning) is searched alongside
    max_depth as an alternative/complementary way to control overfitting.
    """
    depth_cap = max(3, int(np.log2(max(n_samples, 2))) + 2)
    max_depth_options = sorted({d for d in (3, 5, 8, min(12, depth_cap), depth_cap) if d >= 2})
    max_depth_options.append(None)

    grid = {
        "max_depth": max_depth_options,
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "ccp_alpha": [0.0, 0.001, 0.005, 0.01],
    }
    if problem_type == "regression":
        grid["criterion"] = ["squared_error", "absolute_error"]
    else:
        grid["criterion"] = ["gini", "entropy"]
        grid["class_weight"] = [None, "balanced"]
    return grid


def _tune_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, problem_type: str):
    """
    Previously ran fully unconstrained (every sklearn default), which lets
    the tree grow until every leaf is pure/single-sample - memorizing the
    training data. Now searches depth/split/leaf/criterion/ccp_alpha via a
    bounded (grid-or-randomized) cross-validated search.
    """
    cv_folds = _safe_cv_folds(y_train, problem_type)
    param_grid = _decision_tree_param_grid(len(X_train), problem_type)
    estimator = (
        DecisionTreeRegressor(random_state=42) if problem_type == "regression"
        else DecisionTreeClassifier(random_state=42)
    )
    scoring = "r2" if problem_type == "regression" else "accuracy"
    search = _bounded_search(estimator, param_grid, cv=cv_folds, scoring=scoring, max_iter=48)
    search.fit(X_train, y_train)
    return search.best_estimator_, {"tuned_params": search.best_params_}


def _tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, problem_type: str):
    """
    A tuned Random Forest averages many decorrelated trees (bootstrap
    sample + random feature subset per split), which typically beats even a
    well-tuned single Decision Tree on test accuracy - at the cost of the
    single tree's easy visual interpretability. feature_importances_ is
    still exposed, so the existing feature-importance chart works unchanged.
    n_jobs is fixed to 1 on the estimator itself (rather than -1) since the
    outer search already parallelizes across hyperparameter combinations -
    nesting two levels of parallelism would just cause contention.

    n_estimators is capped at 200 and the search is limited to 20 sampled
    combinations (via _bounded_search) specifically because Random Forest is
    the most expensive of the five algorithms to tune - each candidate
    trains many trees, and on a single-core environment an exhaustive grid
    here measured 35-40 seconds for one training request, which is too slow
    for what should feel like a fast, interactive wizard.
    """
    cv_folds = _safe_cv_folds(y_train, problem_type)
    if problem_type == "regression":
        estimator = RandomForestRegressor(random_state=42, n_jobs=1)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": [1.0, "sqrt", "log2"],
        }
        scoring = "r2"
    else:
        estimator = RandomForestClassifier(random_state=42, n_jobs=1)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"],
        }
        scoring = "accuracy"

    search = _bounded_search(estimator, param_grid, cv=cv_folds, scoring=scoring, max_iter=20)
    search.fit(X_train, y_train)
    return search.best_estimator_, {"tuned_params": search.best_params_}


def _reduce_dimensionality_for_knn(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                                    problem_type: str, max_features: int = 20):
    """
    KNN suffers the most of the four (five) algorithms from high-dimensional
    one-hot feature spaces: a high-cardinality categorical column exploded
    into many binary columns dilutes what "nearest" even means (curse of
    dimensionality). When the encoded feature set exceeds max_features,
    keep only the top max_features by mutual information with the target -
    fit on the TRAINING set only, then apply the same selection to the test
    set (no leakage). Below the threshold, features pass through unchanged.
    """
    n_features = X_train.shape[1]
    if n_features <= max_features:
        return X_train, X_test, X_train.columns.tolist(), {}, None

    score_func = mutual_info_regression if problem_type == "regression" else mutual_info_classif
    selector = SelectKBest(score_func=score_func, k=max_features)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    selected_cols = X_train.columns[selector.get_support()].tolist()

    X_train_df = pd.DataFrame(X_train_sel, columns=selected_cols, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_sel, columns=selected_cols, index=X_test.index)

    return X_train_df, X_test_df, selected_cols, {
        "dimensionality_reduced": True,
        "features_before_reduction": n_features,
        "features_after_reduction": max_features,
    }, selector


def _tune_knn(X_train: pd.DataFrame, y_train: pd.Series, problem_type: str, user_k: int, auto_k: bool):
    """
    Jointly tunes n_neighbors, weights ('uniform' vs 'distance'), and p
    (1=Manhattan, 2=Euclidean) via GridSearchCV - previously only k was
    tuned. When auto_k is True, k is searched alongside weights/p (1..20,
    capped to the training set size). When auto_k is False, the user's
    chosen k is kept fixed and only weights/p are tuned around it.

    After finding the best combination, re-runs a k-only CV sweep at the
    winning weights/p so the existing elbow-chart frontend contract
    (k_search / k_search_metric / auto_k / k_used) keeps working unchanged.
    """
    n_samples = len(X_train)
    cv_folds = _safe_cv_folds(y_train, problem_type)
    model_cls = KNeighborsRegressor if problem_type == "regression" else KNeighborsClassifier
    scoring = "r2" if problem_type == "regression" else "accuracy"

    if auto_k:
        k_max = min(20, n_samples - 1)
        k_max = max(k_max, 1)
        k_range = list(range(1, k_max + 1))
    else:
        k_range = [max(1, min(user_k, n_samples - 1))]

    param_grid = {"n_neighbors": k_range, "weights": ["uniform", "distance"], "p": [1, 2]}
    search = _bounded_search(model_cls(), param_grid, cv=cv_folds, scoring=scoring, max_iter=100)
    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_k = best_params["n_neighbors"]

    extras: Dict[str, Any] = {"tuned_params": best_params, "k_used": best_k}

    if auto_k:
        score_label = "accuracy" if problem_type == "classification" else "r2_score"
        k_search_results = []
        for candidate_k in k_range:
            candidate = model_cls(
                n_neighbors=candidate_k, weights=best_params["weights"], p=best_params["p"]
            )
            try:
                scores = cross_val_score(candidate, X_train, y_train, cv=cv_folds, scoring=scoring)
            except ValueError:
                continue
            k_search_results.append({"k": candidate_k, "score": round(float(scores.mean()), 4)})
        extras["k_search"] = k_search_results
        extras["k_search_metric"] = score_label
        extras["auto_k"] = True

    return search.best_estimator_, extras


def _regression_metrics(y_test, y_pred) -> Dict[str, Any]:
    mse = mean_squared_error(y_test, y_pred)
    return {
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
    }


def _regression_extras(X_test, y_test, y_pred) -> Dict[str, Any]:
    """
    If exactly one feature column -> return line-graph-ready points (sorted by x).
    Otherwise -> return actual vs predicted scatter data (generalizes to any feature count).
    """
    extras: Dict[str, Any] = {}
    if X_test.shape[1] == 1:
        feature_name = X_test.columns[0]
        points = sorted(
            zip(X_test[feature_name].tolist(), y_test.tolist(), y_pred.tolist()),
            key=lambda p: p[0]
        )
        extras["chart_type"] = "line_scatter"
        extras["feature_name"] = feature_name
        extras["points"] = [{"x": x, "actual": a, "predicted": p} for x, a, p in points]
    else:
        extras["chart_type"] = "actual_vs_predicted"
        extras["points"] = [
            {"actual": float(a), "predicted": float(p)}
            for a, p in zip(y_test.tolist(), y_pred.tolist())
        ]
    return extras


def _regression_algorithm_extras(algorithm, model, feature_names) -> Dict[str, Any]:
    """Algorithm-specific extras for the regression path."""
    extras: Dict[str, Any] = {}
    if algorithm in TREE_LIKE_ALGORITHMS:
        extras["feature_importances"] = sorted(
            [
                {"feature": f, "importance": round(float(imp), 4)}
                for f, imp in zip(feature_names, model.feature_importances_)
            ],
            key=lambda d: d["importance"], reverse=True
        )
    elif algorithm == "linear_regression":
        # Both RidgeCV and ElasticNetCV expose coef_/intercept_ just like
        # plain LinearRegression did.
        extras["coefficients"] = sorted(
            [
                {"feature": f, "coefficient": round(float(c), 4)}
                for f, c in zip(feature_names, model.coef_)
            ],
            key=lambda d: abs(d["coefficient"]), reverse=True
        )
        extras["intercept"] = round(float(model.intercept_), 4)
    return extras


def _classification_metrics(y_test, y_pred, target_encoder) -> Dict[str, Any]:
    labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )

    if target_encoder is not None:
        label_names = [str(target_encoder.inverse_transform([lbl])[0]) for lbl in labels]
    else:
        label_names = [str(lbl) for lbl in labels]

    per_class = [
        {
            "class": label_names[i],
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    ]

    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": label_names,
            "matrix": cm,
        },
    }


def _probabilistic_classification_info(model, X_test, y_test) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    ROC-AUC and PR-AUC are more informative than accuracy alone under class
    imbalance - exactly the scenario class_weight='balanced' (now searched
    for Logistic Regression / Decision Tree / Random Forest) specifically
    targets. Returns (metric_additions, extra_additions): the AUC numbers
    go into `metrics`, while ROC curve points (binary case only, for a
    chart) go into `extras`. Silently returns ({}, {}) if the model has no
    predict_proba, or if a class present in training never appears in the
    test split (makes AUC ill-defined) - better to omit than crash a
    training run over a metric that isn't the primary result.
    """
    if not hasattr(model, "predict_proba"):
        return {}, {}
    try:
        proba = model.predict_proba(X_test)
        classes = model.classes_
        metric_additions: Dict[str, Any] = {}
        extra_additions: Dict[str, Any] = {}

        if len(classes) == 2:
            pos_idx = list(classes).index(classes[1])
            pos_proba = proba[:, pos_idx]
            metric_additions["roc_auc"] = round(float(roc_auc_score(y_test, pos_proba)), 4)
            metric_additions["pr_auc"] = round(float(average_precision_score(y_test, pos_proba)), 4)
            fpr, tpr, _ = roc_curve(y_test, pos_proba)
            extra_additions["roc_curve"] = [
                {"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)
            ]
        else:
            metric_additions["roc_auc"] = round(
                float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro")), 4
            )
            y_test_bin = label_binarize(y_test, classes=classes)
            metric_additions["pr_auc"] = round(
                float(average_precision_score(y_test_bin, proba, average="macro")), 4
            )
        return metric_additions, extra_additions
    except ValueError:
        return {}, {}


def _classification_extras(algorithm, model, feature_names, X_test, target_encoder) -> Dict[str, Any]:
    extras: Dict[str, Any] = {}

    if algorithm == "logistic_regression":
        coef = model.coef_
        if coef.shape[0] == 1:
            extras["coefficients"] = [
                {"feature": f, "coefficient": round(float(c), 4)}
                for f, c in zip(feature_names, coef[0])
            ]
        else:
            classes = model.classes_
            class_names = (
                [str(target_encoder.inverse_transform([c])[0]) for c in classes]
                if target_encoder is not None else [str(c) for c in classes]
            )
            extras["coefficients_per_class"] = [
                {
                    "class": class_names[i],
                    "coefficients": [
                        {"feature": f, "coefficient": round(float(c), 4)}
                        for f, c in zip(feature_names, coef[i])
                    ],
                }
                for i in range(len(classes))
            ]

    elif algorithm in TREE_LIKE_ALGORITHMS:
        extras["feature_importances"] = sorted(
            [
                {"feature": f, "importance": round(float(imp), 4)}
                for f, imp in zip(feature_names, model.feature_importances_)
            ],
            key=lambda d: d["importance"], reverse=True
        )

    elif algorithm == "knn":
        # Decision boundary plot only makes sense with exactly 2 numeric features
        if X_test.shape[1] == 2:
            extras["decision_boundary_available"] = True
            extras["boundary_features"] = feature_names
            extras["boundary_grid"] = _knn_decision_boundary(model, X_test)
        else:
            extras["decision_boundary_available"] = False

    return extras


def _knn_decision_boundary(model, X_test: pd.DataFrame, resolution: int = 30) -> list:
    """Generate a coarse grid of predictions across the 2D feature space for a decision boundary plot."""
    f1, f2 = X_test.columns[0], X_test.columns[1]
    x_min, x_max = X_test[f1].min() - 1, X_test[f1].max() + 1
    y_min, y_max = X_test[f2].min() - 1, X_test[f2].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid_points = pd.DataFrame({f1: xx.ravel(), f2: yy.ravel()})
    preds = model.predict(grid_points)

    return [
        {"x": float(x), "y": float(y), "predicted_class": int(p)}
        for x, y, p in zip(xx.ravel(), yy.ravel(), preds)
    ]


def _sample_predictions(X_test, y_test, y_pred, target_column, target_encoder, problem_type, n=10) -> list:
    sample = []
    for i in range(min(n, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        if problem_type == "classification" and target_encoder is not None:
            actual = target_encoder.inverse_transform([actual])[0]
            predicted = target_encoder.inverse_transform([predicted])[0]
        sample.append({
            "actual": actual if isinstance(actual, (int, float, str)) else float(actual),
            "predicted": predicted if isinstance(predicted, (int, float, str)) else float(predicted),
        })
    return sample
