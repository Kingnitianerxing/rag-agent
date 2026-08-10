import { useEffect, useState } from "react";
import type { AuthUserSummary, DocumentRecord, SharePayload } from "../../api/types";

const ROLE_OPTIONS = ["editor", "viewer"] as const;

export function ShareDialog({
  doc,
  users,
  onClose,
  onSave,
}: {
  doc: DocumentRecord;
  users: AuthUserSummary[];
  onClose: () => void;
  onSave: (payload: SharePayload) => Promise<void>;
}) {
  const [roles, setRoles] = useState<string[]>(doc.allowed_roles ?? []);
  const [userIds, setUserIds] = useState<number[]>(doc.allowed_user_ids ?? []);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setRoles(doc.allowed_roles ?? []);
    setUserIds(doc.allowed_user_ids ?? []);
  }, [doc]);

  const toggleRole = (role: string) => {
    setRoles((prev) => (prev.includes(role) ? prev.filter((r) => r !== role) : [...prev, role]));
  };

  const toggleUser = (id: number) => {
    setUserIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  const save = async () => {
    setSaving(true);
    try {
      await onSave({ allowed_roles: roles, allowed_user_ids: userIds });
      onClose();
    } finally {
      setSaving(false);
    }
  };

  const shareTargets = users.filter((u) => !u.roles.includes("admin"));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4" role="dialog">
      <div className="w-full max-w-md rounded-lg border border-line bg-surface p-4 shadow-lg">
        <h3 className="font-serif text-lg font-semibold text-ink">Share document</h3>
        <p className="mt-1 truncate font-mono text-xs text-muted" title={doc.source}>
          {doc.source}
        </p>

        <div className="mt-4">
          <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-primary">
            Share with roles
          </p>
          <div className="flex flex-wrap gap-3">
            {ROLE_OPTIONS.map((role) => (
              <label key={role} className="flex items-center gap-2 text-sm text-ink">
                <input
                  type="checkbox"
                  checked={roles.includes(role)}
                  onChange={() => toggleRole(role)}
                />
                {role}
              </label>
            ))}
          </div>
        </div>

        <div className="mt-4">
          <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-primary">
            Share with users
          </p>
          {shareTargets.length === 0 ? (
            <p className="text-sm text-muted">No editor/viewer users available.</p>
          ) : (
            <ul className="max-h-40 space-y-2 overflow-y-auto rounded border border-line p-2">
              {shareTargets.map((u) => (
                <li key={u.id}>
                  <label className="flex items-center gap-2 text-sm text-ink">
                    <input
                      type="checkbox"
                      checked={userIds.includes(u.id)}
                      onChange={() => toggleUser(u.id)}
                    />
                    <span>{u.username}</span>
                    <span className="text-xs text-muted">({u.roles.join(", ") || "none"})</span>
                  </label>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            className="rounded border border-line px-3 py-1.5 text-sm text-muted hover:bg-sunken"
            onClick={onClose}
            disabled={saving}
          >
            Cancel
          </button>
          <button
            type="button"
            className="rounded bg-primary px-3 py-1.5 text-sm text-white hover:opacity-90 disabled:opacity-50"
            onClick={() => void save()}
            disabled={saving}
          >
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
