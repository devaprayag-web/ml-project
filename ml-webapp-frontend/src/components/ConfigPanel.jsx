const ALGORITHMS = [
  { value: "linear_regression", label: "Linear Regression", type: "regression" },
  { value: "logistic_regression", label: "Logistic Regression", type: "classification" },
  { value: "knn", label: "K-Nearest Neighbors", type: "auto-detected" },
  { value: "decision_tree", label: "Decision Tree", type: "auto-detected" },
  { value: "random_forest", label: "Random Forest", type: "auto-detected" },
];

export default function ConfigPanel({
  columns,
  targetColumn,
  onTargetColumnChange,
  algorithm,
  onAlgorithmChange,
  k,
  onKChange,
  autoK,
  onAutoKChange,
  onTrain,
  isTraining,
  error,
}) {
  return (
    <div className="panel">
      <div className="field">
        <label htmlFor="target-column">Target column (what to predict)</label>
        <select
          id="target-column"
          value={targetColumn}
          onChange={(e) => onTargetColumnChange(e.target.value)}
        >
          <option value="">Select a column&hellip;</option>
          {columns.map((col) => (
            <option key={col.name} value={col.name}>
              {col.name} ({col.dtype})
            </option>
          ))}
        </select>
        <p className="field-hint">
          Linear Regression needs a numeric column, and Logistic Regression needs a
          small set of discrete classes. KNN, Decision Tree, and Random Forest work
          with either &mdash; they automatically train as regression or classification
          based on your target column.
        </p>
      </div>

      <div className="field">
        <label>Algorithm</label>
        <div className="algo-options">
          {ALGORITHMS.map((algo) => (
            <label
              key={algo.value}
              className={`algo-option ${algorithm === algo.value ? "selected" : ""}`}
            >
              <input
                type="radio"
                name="algorithm"
                value={algo.value}
                checked={algorithm === algo.value}
                onChange={() => onAlgorithmChange(algo.value)}
              />
              {algo.label}
              <span className="algo-type">{algo.type}</span>
            </label>
          ))}
        </div>
      </div>

      {algorithm === "knn" && (
        <div className="field" style={{ maxWidth: 280 }}>
          <label className="checkbox-label" style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 400 }}>
            <input
              type="checkbox"
              checked={autoK}
              onChange={(e) => onAutoKChange(e.target.checked)}
            />
            Auto-select best k (elbow method)
          </label>
          <p className="field-hint">
            Tries k = 1&ndash;20 using cross-validation on the training data and picks
            the best-scoring one (accuracy for classification, R² for regression).
            Turn this off to set k manually.
          </p>

          <label htmlFor="k-value" style={{ marginTop: 12, display: "block" }}>
            k (neighbors)
          </label>
          <input
            id="k-value"
            type="number"
            min={1}
            max={50}
            value={k}
            disabled={autoK}
            onChange={(e) => onKChange(Number(e.target.value))}
          />
        </div>
      )}

      {error && <div className="notice notice-error">{error}</div>}

      <button
        className="btn"
        onClick={onTrain}
        disabled={!targetColumn || !algorithm || isTraining}
      >
        {isTraining ? "Training…" : "Train model"}
      </button>
    </div>
  );
}
