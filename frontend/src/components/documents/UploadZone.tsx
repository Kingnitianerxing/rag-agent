import { UploadCloud } from "lucide-react";
import { useRef, useState } from "react";

const ALLOWED = [".pdf", ".md", ".markdown", ".png", ".jpg", ".jpeg", ".webp"];

function allowed(name: string): boolean {
  return ALLOWED.some((s) => name.toLowerCase().endsWith(s));
}

export function UploadZone({ onFile }: { onFile: (f: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [error, setError] = useState("");

  const handle = (file: File | undefined) => {
    if (!file) return;
    if (!allowed(file.name)) {
      setError(`Unsupported file type. Allowed: ${ALLOWED.join(", ")}`);
      return;
    }
    setError("");
    onFile(file);
  };

  return (
    <div>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          handle(e.dataTransfer.files[0]);
        }}
        className={`flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed p-8 text-sm transition ${
          over ? "border-primary bg-primary/5" : "border-muted/50 text-muted"
        }`}
      >
        <UploadCloud size={28} />
        Drop a PDF, Markdown, or image (PNG/JPG/WEBP) here, or click to choose
        <input
          ref={inputRef}
          data-testid="file-input"
          type="file"
          accept={ALLOWED.join(",")}
          className="hidden"
          onChange={(e) => handle(e.target.files?.[0])}
        />
      </div>
      {error && <p className="mt-2 text-sm text-danger">{error}</p>}
    </div>
  );
}
