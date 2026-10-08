/**
 * Renders one numbered step in the vertical workflow (Upload -> Configure -> Results).
 * status: "done" | "active" | "upcoming"
 */
export default function Step({ number, status, title, subtitle, isLast, children }) {
  return (
    <div className="step">
      <div className="step-marker-col">
        <div className={`step-number ${status === "active" ? "active" : ""} ${status === "done" ? "done" : ""}`}>
          {status === "done" ? "✓" : number}
        </div>
        {!isLast && <div className="step-connector" />}
      </div>
      <div className="step-body">
        <div className="step-title">{title}</div>
        {subtitle && <div className="step-subtitle">{subtitle}</div>}
        {status !== "upcoming" && children}
      </div>
    </div>
  );
}
