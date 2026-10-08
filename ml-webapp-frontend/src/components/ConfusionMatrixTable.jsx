export default function ConfusionMatrixTable({ labels, matrix }) {
  const maxVal = Math.max(...matrix.flat());

  function cellStyle(rowIdx, colIdx, value) {
    const isCorrect = rowIdx === colIdx;
    const intensity = maxVal > 0 ? value / maxVal : 0;
    return {
      background: isCorrect
        ? `rgba(14, 92, 86, ${0.15 + intensity * 0.5})`
        : intensity > 0
        ? `rgba(178, 59, 59, ${0.1 + intensity * 0.35})`
        : "transparent",
      fontWeight: isCorrect ? 600 : 400,
    };
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 4 }}>
        <span className="cm-axis-label">predicted &rarr;</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
        <span
          className="cm-axis-label"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          actual &darr;
        </span>
        <div className="confusion-matrix">
          <table>
            <thead>
              <tr>
                <th className="cm-header"></th>
                {labels.map((label) => (
                  <th key={label} className="cm-header">{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  <th className="cm-header">{labels[rowIdx]}</th>
                  {row.map((value, colIdx) => (
                    <td key={colIdx} style={cellStyle(rowIdx, colIdx, value)}>
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
