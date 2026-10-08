# No-Code ML Trainer — Frontend

React + Vite frontend for the No-Code ML Trainer. Talks to the FastAPI
backend running at `http://127.0.0.1:8000`.

## Setup

Make sure the backend is running first (see the backend project's README).

```bash
npm install
npm run dev
```

Open the URL it prints (usually `http://localhost:5173`).

## How it's organized

```
src/
├── App.jsx                      # top-level state machine: upload -> configure -> results
├── api/client.js                # all fetch() calls to the backend, in one place
└── components/
    ├── Step.jsx                 # numbered step wrapper (the 1/2/3 workflow markers)
    ├── FileUpload.jsx           # drag-and-drop CSV upload
    ├── DatasetPreview.jsx       # column info table + sample rows after upload
    ├── ConfigPanel.jsx          # target column + algorithm + k selection
    ├── ResultsDashboard.jsx     # routes to the right charts based on problem_type/algorithm
    └── ConfusionMatrixTable.jsx # heatmap-style confusion matrix
```

## How results routing works

`ResultsDashboard.jsx` is the piece worth understanding before you extend
this — it branches on two things from the `/api/train` response:

1. **`problem_type`** (`"regression"` vs `"classification"`) — decides
   whether you're in `RegressionResults` or `ClassificationResults`.
2. **`extras.chart_type`** (regression only) — `"line_scatter"` for a
   single-feature dataset, `"actual_vs_predicted"` for everything else.
3. **`algorithm`** (classification only) — decides which extra chart
   renders below the confusion matrix: coefficients (logistic regression),
   feature importances (decision tree), or a decision boundary plot (KNN,
   only when exactly 2 features remain).

If you add a 5th algorithm later, this is the file to extend — add a new
branch here rather than scattering conditional rendering elsewhere.

## Known limitations / good next features

- No loading skeleton — just a plain text "Training…" message. A proper
  skeleton loader would look more polished for a portfolio demo.
- No "compare algorithms side by side" view yet — currently one algorithm
  at a time. Backend already returns everything needed per algorithm to
  build this as a v2 feature (run several `/api/train` calls, render a
  comparison table).
- No dark mode.
- Session state lives only in React state — refreshing the page loses your
  uploaded dataset (matches the backend's in-memory session store, which
  has the same limitation).
