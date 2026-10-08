import { useRef, useState } from "react";

export default function FileUpload({ onFileSelected, disabled }) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  function handleFiles(files) {
    const file = files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      alert("Please select a .csv file.");
      return;
    }
    onFileSelected(file);
  }

  return (
    <div
      className={`dropzone ${isDragging ? "dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
    >
      <p>Drag a .csv file here, or</p>
      <button
        type="button"
        className="btn btn-secondary"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
      >
        Choose file
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
