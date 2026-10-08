import { Suspense, lazy, useState } from "react";
import Step from "./components/Step";
import FileUpload from "./components/FileUpload";
import DatasetPreview from "./components/DatasetPreview";
import ConfigPanel from "./components/ConfigPanel";
import { uploadCsv, trainModel } from "./api/client";
import "./App.css";

const ResultsDashboard = lazy(() => import("./components/ResultsDashboard"));

export default function App() {
  const [uploadResult, setUploadResult] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const [targetColumn, setTargetColumn] = useState("");
  const [algorithm, setAlgorithm] = useState("logistic_regression");
  const [k, setK] = useState(5);
  const [autoK, setAutoK] = useState(false);

  const [trainResult, setTrainResult] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainError, setTrainError] = useState(null);

  async function handleFileSelected(file) {
    setIsUploading(true);
    setUploadError(null);
    setUploadResult(null);
    setTrainResult(null);
    setTargetColumn("");
    try {
      const result = await uploadCsv(file);
      setUploadResult(result);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setIsUploading(false);
    }
  }

  async function handleTrain() {
    setIsTraining(true);
    setTrainError(null);
    setTrainResult(null);
    try {
      const result = await trainModel({
        sessionId: uploadResult.session_id,
        algorithm,
        targetColumn,
        k,
        autoK,
      });
      setTrainResult(result);
    } catch (err) {
      setTrainError(err.message);
    } finally {
      setIsTraining(false);
    }
  }

  const uploadStatus = uploadResult ? "done" : "active";
  const configStatus = !uploadResult ? "upcoming" : trainResult ? "done" : "active";
  const resultsStatus = trainResult ? "active" : "upcoming";

  return (
    <div className="app">
      <div className="masthead">
        <h1>No-Code ML Trainer</h1>
        <p>
          Upload a CSV, pick what you want to predict, and see how Linear Regression,
          Logistic Regression, KNN, and Decision Tree each handle it — nulls filled and
          categorical columns encoded automatically.
        </p>
      </div>

      <Step
        number={1}
        status={uploadStatus}
        title="Upload your dataset"
        subtitle="CSV files only. Nulls and text columns are fine — they're handled automatically."
      >
        <FileUpload onFileSelected={handleFileSelected} disabled={isUploading} />
        {isUploading && <p className="loading-text" style={{ marginTop: 12 }}>Uploading &amp; parsing…</p>}
        {uploadError && <div className="notice notice-error" style={{ marginTop: 12 }}>{uploadError}</div>}
        {uploadResult && (
          <div style={{ marginTop: 16 }}>
            <DatasetPreview uploadResult={uploadResult} />
          </div>
        )}
      </Step>

      <Step
        number={2}
        status={configStatus}
        title="Choose target and algorithm"
        subtitle="Pick the column you want to predict, and which algorithm to train."
      >
        {uploadResult && (
          <ConfigPanel
            columns={uploadResult.columns}
            targetColumn={targetColumn}
            onTargetColumnChange={setTargetColumn}
            algorithm={algorithm}
            onAlgorithmChange={setAlgorithm}
            k={k}
            onKChange={setK}
            autoK={autoK}
            onAutoKChange={setAutoK}
            onTrain={handleTrain}
            isTraining={isTraining}
            error={trainError}
          />
        )}
      </Step>

      <Step number={3} status={resultsStatus} title="Results" isLast>
        {isTraining && <p className="loading-text">Training model&hellip;</p>}
        {trainResult && (
          <Suspense fallback={<p className="loading-text">Loading results…</p>}>
            <ResultsDashboard result={trainResult} />
          </Suspense>
        )}
      </Step>
    </div>
  );
}
