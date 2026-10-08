// Reads from a .env file (VITE_API_BASE_URL=https://your-api.example.com) so the
// deployed build points at the real backend instead of localhost. Falls back to
// the local dev server when the variable isn't set.
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

/**
 * Uploads a CSV file to the backend.
 * Returns { session_id, row_count, columns, preview_rows, dropped_columns }
 */
export async function uploadCsv(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Upload failed.");
  }
  return data;
}

/**
 * Trains the selected algorithm on the uploaded dataset.
 * Returns { problem_type, algorithm, metrics, extras, sample_predictions, model_id }
 */
export async function trainModel({ sessionId, algorithm, targetColumn, k, autoK }) {
  const res = await fetch(`${API_BASE}/api/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      algorithm,
      target_column: targetColumn,
      k,
      auto_k: autoK,
    }),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Training failed.");
  }
  return data;
}

/**
 * Downloads the trained model (fitted estimator + preprocessor + scaler,
 * bundled with joblib on the backend) as a .joblib file and triggers a
 * browser save-as via a temporary anchor element.
 */
export async function exportModel(modelId) {
  const res = await fetch(`${API_BASE}/api/export/${modelId}`);
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Export failed.");
  }
  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="([^"]+)"/);
  const filename = match ? match[1] : `model_${modelId}.joblib`;

  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}
