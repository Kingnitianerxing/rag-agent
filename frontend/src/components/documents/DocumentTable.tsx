import { Share2, Trash2 } from "lucide-react";
import type { DocumentRecord } from "../../api/types";

function shareSummary(d: DocumentRecord): string {
  const roles = d.allowed_roles ?? [];
  const users = d.allowed_user_ids ?? [];
  if (roles.length === 0 && users.length === 0) return "Private";
  const parts: string[] = [];
  if (roles.length) parts.push(`roles: ${roles.join(", ")}`);
  if (users.length) parts.push(`${users.length} user(s)`);
  return parts.join(" · ");
}

export function DocumentTable({
  docs,
  onDelete,
  onShare,
}: {
  docs: DocumentRecord[];
  onDelete: (id: string) => void;
  onShare?: (doc: DocumentRecord) => void;
}) {
  if (docs.length === 0) return <p className="text-sm text-muted">No documents ingested yet.</p>;
  return (
    <table className="w-full text-left text-sm">
      <thead className="text-xs uppercase text-muted">
        <tr>
          <th className="py-2">Source</th>
          <th className="py-2">Type</th>
          <th className="py-2">Chunks</th>
          <th className="py-2">Shared</th>
          <th className="py-2">Ingested</th>
          <th className="py-2" />
        </tr>
      </thead>
      <tbody>
        {docs.map((d) => (
          <tr key={d.id} className="border-t border-muted/20">
            <td className="max-w-xs truncate py-2 font-mono text-xs">{d.source}</td>
            <td className="py-2 text-xs text-muted">
              {d.modality === "image" ? "Image" : "Text"}
            </td>
            <td className="py-2">{d.chunks}</td>
            <td className="py-2 text-xs text-muted">{shareSummary(d)}</td>
            <td className="py-2 text-muted">{d.ingested_at}</td>
            <td className="py-2 text-right">
              <div className="inline-flex items-center gap-2">
                {d.can_share && onShare && (
                  <button
                    aria-label={`Share ${d.source}`}
                    className="text-muted hover:text-primary"
                    onClick={() => onShare(d)}
                  >
                    <Share2 size={16} />
                  </button>
                )}
                {d.can_delete !== false && (
                  <button
                    aria-label={`Delete ${d.source}`}
                    className="text-muted hover:text-danger"
                    onClick={() => onDelete(d.id)}
                  >
                    <Trash2 size={16} />
                  </button>
                )}
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
