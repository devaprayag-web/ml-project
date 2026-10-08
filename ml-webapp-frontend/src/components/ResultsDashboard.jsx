import { useState } from "react";
import {
  ResponsiveContainer, ComposedChart, ScatterChart, BarChart, LineChart,
  Scatter, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, Cell,
} from "recharts";
import ConfusionMatrixTable from "./ConfusionMatrixTable";
import { exportModel } from "../api/client";

const TEAL = "#0E5C56";
const OCHRE = "#C97A2B";
const INK = "#1C2321";

export default function ResultsDashboard({ result }) {
  const { problem_type, algorithm, metrics, extras, sample_predictions, model_id } = result;

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
        <ExportModelButton modelId={model_id} />
      </div>
      <TunedParamsBadge extras={extras} />
      <PreprocessingNotice extras={extras} />

      {problem_type === "regression" ? (
        <RegressionResults metrics={metrics} extras={extras} algorithm={algorithm} />
      ) : (
        <ClassificationResults metrics={metrics} extras={extras} algorithm={algorithm} />
      )}

      <CvValidationNote extras={extras} />

      <div className="section-heading">Sample predictions</div>
      <div className="scroll-x">
        <table className="data-table">
          <thead>
            <tr>
              <th>actual</th>
              <th>predicted</th>
            </tr>
          </thead>
          <tbody>
            {sample_predictions.map((row, i) => (
              <tr key={i}>
                <td>{String(row.actual)}</td>
                <td style={{ color: row.actual === row.predicted ? "inherit" : "var(--danger)" }}>
                  {String(row.predicted)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Downloads the trained model (estimator + preprocessor + scaler, bundled
// via joblib on the backend) so it can be reused outside this app.
function ExportModelButton({ modelId }) {
  const [state, setState] = useState("idle"); // idle | exporting | error
  const [error, setError] = useState(null);

  async function handleClick() {
    setState("exporting");
    setError(null);
    try {
      await exportModel(modelId);
      setState("idle");
    } catch (err) {
      setError(err.message);
      setState("error");
    }
  }

  if (!modelId) return null;

  return (
    <div style={{ textAlign: "right" }}>
      <button className="btn btn-secondary" onClick={handleClick} disabled={state === "exporting"}>
        {state === "exporting" ? "Exporting…" : "Export model (.joblib)"}
      </button>
      {error && <div className="notice notice-error" style={{ marginTop: 8 }}>{error}</div>}
    </div>
  );
}

// Surfaces the preprocessing steps that ran silently before training:
// missingness indicators added, outliers capped, dimensionality reduced.
// Only renders anything that actually happened for this dataset.
function PreprocessingNotice({ extras }) {
  const notes = [];
  if (extras.missingness_indicators_added?.length > 0) {
    notes.push(
      `Added a "was missing" flag for: ${extras.missingness_indicators_added.map((c) => c.replace(/_was_missing$/, "")).join(", ")}`
    );
  }
  if (extras.outliers_capped && Object.keys(extras.outliers_capped).length > 0) {
    const parts = Object.entries(extras.outliers_capped).map(([col, n]) => `${col} (${n})`);
    notes.push(`Capped extreme outlier values in: ${parts.join(", ")}`);
  }
  if (extras.dimensionality_reduced) {
    notes.push(
      `Reduced ${extras.features_before_reduction} features down to the ${extras.features_after_reduction} most informative for distance-based matching`
    );
  }
  if (extras.scaling_skipped_reason === "all_features_binary") {
    notes.push("Skipped feature scaling — every feature is already binary (0/1)");
  }
  if (notes.length === 0) return null;
  return (
    <div className="notice notice-info" style={{ marginBottom: 16 }}>
      {notes.map((n) => (
        <div key={n}>{n}</div>
      ))}
    </div>
  );
}

// Shows the k-fold cross-validation summary computed on the training set
// with the winning hyperparameters — a more trustworthy generalization
// estimate than the single 80/20 test split alone, shown alongside it.
function CvValidationNote({ extras }) {
  const cv = extras.cv_validation;
  if (!cv) return null;
  const label = cv.metric === "r2_score" ? "R\u00b2" : "accuracy";
  return (
    <p className="field-hint" style={{ marginTop: -4, marginBottom: 20 }}>
      Cross-validated on the training data ({cv.folds}-fold): {label} ={" "}
      <strong>{cv.mean}</strong> ± {cv.std} — a more stable estimate than the single test split above.
    </p>
  );
}

const PARAM_LABELS = {
  n_neighbors: "k",
  weights: "weights",
  p: "distance metric",
  C: "C (regularization)",
  class_weight: "class weight",
  max_depth: "max depth",
  min_samples_split: "min samples split",
  min_samples_leaf: "min samples leaf",
  criterion: "criterion",
};

function formatParamValue(key, value) {
  if (value === null) return key === "max_depth" ? "unlimited" : "none";
  if (key === "p") return value === 1 ? "Manhattan" : "Euclidean";
  return String(value);
}

// Shows what GridSearchCV / RidgeCV actually picked during training, so the
// tuning that happens behind the scenes (previously invisible) is visible.
function TunedParamsBadge({ extras }) {
  const hasTunedParams = extras.tuned_params && Object.keys(extras.tuned_params).length > 0;
  const hasAlpha = extras.regularization_alpha !== undefined;
  if (!hasTunedParams && !hasAlpha) return null;

  const entries = hasTunedParams
    ? Object.entries(extras.tuned_params).map(([key, value]) => [
        PARAM_LABELS[key] || key,
        formatParamValue(key, value),
      ])
    : [["regularization strength (alpha)", extras.regularization_alpha]];

  return (
    <div className="tuned-params-badge">
      <span className="tuned-params-title">Tuned automatically:</span>
      {entries.map(([label, value]) => (
        <span key={label} className="tuned-params-chip">
          {label}: <strong>{value}</strong>
        </span>
      ))}
    </div>
  );
}

function KnnElbowChart({ extras }) {
  if (!extras.auto_k || !extras.k_search) return null;
  const metricLabel = extras.k_search_metric === "r2_score" ? "R\u00b2 score" : "Accuracy";
  return (
    <>
      <div className="section-heading">
        {metricLabel} vs k &mdash; best k = {extras.k_used}
      </div>
      <p className="field-hint" style={{ marginTop: -8, marginBottom: 12 }}>
        Each point is the average {metricLabel.toLowerCase()} across cross-validation folds
        on the training data for that k. The model above was trained using the k with
        the best score.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={extras.k_search} margin={{ left: 8, right: 16 }}>
          <CartesianGrid stroke="#DAD6CC" />
          <XAxis dataKey="k" tick={{ fontSize: 11 }} label={{ value: "k", position: "insideBottom", offset: -4, fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} domain={["dataMin - 0.02", "dataMax + 0.02"]} />
          <Tooltip contentStyle={{ fontSize: 12, fontFamily: "var(--font-mono)" }} />
          <ReferenceLine x={extras.k_used} stroke={OCHRE} strokeDasharray="4 4" />
          <Line type="monotone" dataKey="score" stroke={TEAL} dot={{ r: 3 }} />
        </LineChart>
      </ResponsiveContainer>
    </>
  );
}

function RegressionResults({ metrics, extras, algorithm }) {
  return (
    <>
      <div className="metrics-grid">
        <MetricBox label="R² score" value={metrics.r2_score} />
        <MetricBox label="RMSE" value={metrics.rmse} />
        <MetricBox label="MAE" value={metrics.mae} />
        <MetricBox label="MSE" value={metrics.mse} />
      </div>

      <div className="section-heading">
        {extras.chart_type === "line_scatter" ? "Fit line vs. actual data" : "Actual vs. predicted"}
      </div>
      {extras.chart_type === "line_scatter" ? (
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart data={extras.points} margin={{ left: 8, right: 16 }}>
            <CartesianGrid stroke="#DAD6CC" />
            <XAxis dataKey="x" type="number" name={extras.feature_name} tick={{ fontSize: 11 }} />
            <YAxis type="number" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ fontSize: 12, fontFamily: "var(--font-mono)" }} />
            <Scatter dataKey="actual" fill={INK} name="Actual" />
            <Line dataKey="predicted" stroke={TEAL} strokeWidth={2} dot={false} name="Predicted" />
          </ComposedChart>
        </ResponsiveContainer>
      ) : (
        <ResponsiveContainer width="100%" height={320}>
          <ScatterChart margin={{ left: 8, right: 16 }}>
            <CartesianGrid stroke="#DAD6CC" />
            <XAxis dataKey="actual" type="number" name="Actual" tick={{ fontSize: 11 }} />
            <YAxis dataKey="predicted" type="number" name="Predicted" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ fontSize: 12, fontFamily: "var(--font-mono)" }} cursor={{ strokeDasharray: "3 3" }} />
            <ReferenceLine
              segment={diagonalSegment(extras.points)}
              stroke={OCHRE}
              strokeDasharray="4 4"
            />
            <Scatter data={extras.points} fill={TEAL} />
          </ScatterChart>
        </ResponsiveContainer>
      )}
      <p className="field-hint">
        {extras.chart_type === "actual_vs_predicted" &&
          "The dashed line is a perfect prediction (actual = predicted) — the closer the dots sit to it, the better the model."}
      </p>

      {(algorithm === "decision_tree" || algorithm === "random_forest") && extras.feature_importances && (
        <>
          <div className="section-heading">Feature importance</div>
          <HorizontalBarChart
            data={extras.feature_importances}
            dataKey="importance"
            labelKey="feature"
            color={OCHRE}
          />
        </>
      )}

      {algorithm === "knn" && <KnnElbowChart extras={extras} />}
    </>
  );
}

function ClassificationResults({ metrics, extras, algorithm }) {
  return (
    <>
      <div className="metrics-grid">
        <MetricBox label="Accuracy" value={metrics.accuracy} />
        {metrics.roc_auc !== undefined && <MetricBox label="ROC-AUC" value={metrics.roc_auc} />}
        {metrics.pr_auc !== undefined && <MetricBox label="PR-AUC" value={metrics.pr_auc} />}
      </div>

      <div className="section-heading">Per-class metrics</div>
      <p className="field-hint" style={{ marginTop: -8, marginBottom: 12 }}>
        Check recall on minority classes closely — high overall accuracy can hide a model that rarely detects the class you actually care about.
      </p>
      <div className="scroll-x">
        <table className="data-table">
          <thead>
            <tr>
              <th>class</th>
              <th>precision</th>
              <th>recall</th>
              <th>f1</th>
              <th>support</th>
            </tr>
          </thead>
          <tbody>
            {metrics.per_class.map((row) => (
              <tr key={row.class}>
                <td style={{ fontWeight: 500 }}>{row.class}</td>
                <td>{row.precision}</td>
                <td style={{ color: row.recall < 0.3 ? "var(--danger)" : "inherit" }}>{row.recall}</td>
                <td>{row.f1_score}</td>
                <td>{row.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="section-heading">Confusion matrix</div>
      <ConfusionMatrixTable
        labels={metrics.confusion_matrix.labels}
        matrix={metrics.confusion_matrix.matrix}
      />

      {extras.roc_curve && (
        <>
          <div className="section-heading">ROC curve (ROC-AUC = {metrics.roc_auc})</div>
          <p className="field-hint" style={{ marginTop: -8, marginBottom: 12 }}>
            The closer the curve hugs the top-left corner, the better the model separates the two classes at every threshold — the dashed diagonal is what a random guess would look like.
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={extras.roc_curve} margin={{ left: 8, right: 16 }}>
              <CartesianGrid stroke="#DAD6CC" />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: "False positive rate", position: "insideBottom", offset: -4, fontSize: 11 }} />
              <YAxis dataKey="tpr" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ fontSize: 12, fontFamily: "var(--font-mono)" }} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={OCHRE} strokeDasharray="4 4" />
              <Line type="monotone" dataKey="tpr" stroke={TEAL} dot={false} strokeWidth={2} />
            </ComposedChart>
          </ResponsiveContainer>
        </>
      )}

      {algorithm === "logistic_regression" && extras.coefficients && (
        <>
          <div className="section-heading">Feature coefficients</div>
          <p className="field-hint" style={{ marginTop: -8, marginBottom: 12 }}>
            Positive values push toward the second class; negative values push toward the first.
          </p>
          <HorizontalBarChart
            data={extras.coefficients}
            dataKey="coefficient"
            labelKey="feature"
          />
        </>
      )}

      {(algorithm === "decision_tree" || algorithm === "random_forest") && extras.feature_importances && (
        <>
          <div className="section-heading">Feature importance</div>
          <HorizontalBarChart
            data={extras.feature_importances}
            dataKey="importance"
            labelKey="feature"
            color={OCHRE}
          />
        </>
      )}

      {algorithm === "knn" && (
        <>
          <KnnElbowChart extras={extras} />

          <div className="section-heading">Decision boundary</div>
          {extras.decision_boundary_available ? (
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ left: 8, right: 16 }}>
                <CartesianGrid stroke="#DAD6CC" />
                <XAxis dataKey="x" type="number" name={extras.boundary_features[0]} tick={{ fontSize: 11 }} />
                <YAxis dataKey="y" type="number" name={extras.boundary_features[1]} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ fontSize: 12, fontFamily: "var(--font-mono)" }} />
                <Scatter data={extras.boundary_grid}>
                  {extras.boundary_grid.map((point, i) => (
                    <Cell key={i} fill={point.predicted_class === 0 ? "#B7D3CF" : "#EBC9A6"} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <p className="field-hint">
              Only available when exactly 2 feature columns remain after preprocessing — this dataset has more,
              so the confusion matrix above is the primary result for KNN here.
            </p>
          )}
        </>
      )}
    </>
  );
}

function MetricBox({ label, value }) {
  return (
    <div className="metric-box">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

function HorizontalBarChart({ data, dataKey, labelKey, color = TEAL }) {
  const sorted = [...data].sort((a, b) => Math.abs(b[dataKey]) - Math.abs(a[dataKey]));
  return (
    <ResponsiveContainer width="100%" height={Math.max(200, sorted.length * 32)}>
      <BarChart data={sorted} layout="vertical" margin={{ left: 24, right: 16 }}>
        <CartesianGrid stroke="#DAD6CC" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11 }} />
        <YAxis dataKey={labelKey} type="category" width={140} tick={{ fontSize: 11 }} />
        <Tooltip contentStyle={{ fontSize: 12, fontFamily: "var(--font-mono)" }} />
        <Bar dataKey={dataKey} fill={color} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function diagonalSegment(points) {
  const all = points.flatMap((p) => [p.actual, p.predicted]);
  const min = Math.min(...all);
  const max = Math.max(...all);
  return [{ x: min, y: min }, { x: max, y: max }];
}
