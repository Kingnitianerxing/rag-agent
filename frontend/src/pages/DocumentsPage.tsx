import { useCallback, useEffect, useState } from "react";
import { ApiError, getJson } from "../api/client";
import type { AuthUserSummary, DocumentRecord } from "../api/types";
import { DocumentTable } from "../components/documents/DocumentTable";
import { ShareDialog } from "../components/documents/ShareDialog";
import { UploadZone } from "../components/documents/UploadZone";
import { UrlIngest } from "../components/documents/UrlIngest";
import { useAuth } from "../context/AuthContext";
import { useSettings } from "../context/SettingsContext";
import { useToast } from "../context/ToastContext";
import { useDocuments } from "../hooks/useDocuments";

export function DocumentsPage() {
  const { docs, refresh, remove, share, ingestUrl, ingestFile } = useDocuments();
  const { canIngest, isAdmin, user, loading } = useAuth();
  const { client } = useSettings();
  const { toast } = useToast();
  const [shareDoc, setShareDoc] = useState<DocumentRecord | null>(null);
  const [users, setUsers] = useState<AuthUserSummary[]>([]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const openShare = useCallback(
    async (doc: DocumentRecord) => {
      try {
        const list = await getJson<AuthUserSummary[]>(client, "/auth/users");
        setUsers(list);
        setShareDoc(doc);
      } catch (e) {
        toast(e instanceof ApiError ? `${e.status}: ${e.detail}` : "Failed to load users", "error");
      }
    },
    [client, toast],
  );

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-6 px-4 py-8">
        <h1 className="font-serif text-2xl font-semibold text-ink">Documents</h1>
        {canIngest ? (
          <section className="rounded-lg border border-line bg-surface p-4 shadow-sm">
            <h2 className="mb-3 text-[11px] font-medium uppercase tracking-wider text-primary">
              Add documents
            </h2>
            <UploadZone onFile={ingestFile} />
            <div className="mt-4">
              <UrlIngest onSubmit={ingestUrl} />
            </div>
          </section>
        ) : (
          !loading &&
          user && (
            <p className="rounded-lg border border-line bg-sunken px-4 py-3 text-sm text-muted">
              Your role ({user.roles.join(", ") || "none"}) cannot ingest documents. Sign in as
              editor or admin to upload.
            </p>
          )
        )}
        <section className="rounded-lg border border-line bg-surface p-4 shadow-sm">
          <h2 className="mb-3 text-[11px] font-medium uppercase tracking-wider text-primary">
            Ingested documents
          </h2>
          <DocumentTable
            docs={docs}
            onDelete={remove}
            onShare={isAdmin ? openShare : undefined}
          />
        </section>
      </div>
      {shareDoc && (
        <ShareDialog
          doc={shareDoc}
          users={users}
          onClose={() => setShareDoc(null)}
          onSave={(payload) => share(shareDoc.id, payload)}
        />
      )}
    </div>
  );
}
