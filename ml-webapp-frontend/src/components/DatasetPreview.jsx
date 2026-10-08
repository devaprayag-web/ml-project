export default function DatasetPreview({ uploadResult }) {
  const { row_count, columns, preview_rows, dropped_columns } = uploadResult;

  return (
    <div className="panel">
      <p style={{ fontSize: 13, color: "var(--slate)", marginTop: 0, marginBottom: 16 }}>
        <strong style={{ color: "var(--ink)" }}>{row_count.toLocaleString()}</strong> rows &middot;{" "}
        <strong style={{ color: "var(--ink)" }}>{columns.length}</strong> columns
      </p>

      {dropped_columns.length > 0 && (
        <div className="notice">
          Dropped {dropped_columns.length} column(s) that were entirely empty:{" "}
          <span className="mono">{dropped_columns.join(", ")}</span>
        </div>
      )}

      <div className="scroll-x">
        <table className="data-table">
          <thead>
            <tr>
              <th>column</th>
              <th>dtype</th>
              <th>nulls</th>
              <th>unique</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((col) => (
              <tr key={col.name}>
                <td style={{ fontWeight: 500 }}>{col.name}</td>
                <td>{col.dtype}</td>
                <td style={{ color: col.null_count > 0 ? "var(--ochre)" : "inherit" }}>
                  {col.null_count}
                </td>
                <td>{col.unique_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {dropped_columns.length === 0 && Object.values(columns).some(c => c.null_count > 0) && (
        <p className="field-hint" style={{ marginTop: 12 }}>
          Nulls will be auto-filled during training — median for numeric columns, most frequent value for text columns.
        </p>
      )}

      <div className="section-heading" style={{ marginTop: 24 }}>Preview (first 5 rows)</div>
      <div className="scroll-x">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col.name}>{col.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview_rows.map((row, i) => (
              <tr key={i}>
                {columns.map((col) => (
                  <td key={col.name}>
                    {row[col.name] === null || row[col.name] === undefined ? (
                      <span style={{ color: "var(--ochre)" }}>null</span>
                    ) : (
                      String(row[col.name])
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
