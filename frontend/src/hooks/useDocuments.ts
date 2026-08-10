import { useCallback, useState } from "react";
import { ApiError, del, getJson, patchJson, postJson, uploadFile } from "../api/client";
import type { DocumentRecord, IngestResult, SharePayload } from "../api/types";
import { useSettings } from "../context/SettingsContext";
import { useToast } from "../context/ToastContext";

export function useDocuments() {
  const { client } = useSettings();
  const { toast } = useToast();
  const [docs, setDocs] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(false);

  const fail = useCallback(
    (e: unknown) => toast(e instanceof ApiError ? `${e.status}: ${e.detail}` : "Request failed", "error"),
    [toast],
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setDocs(await getJson<DocumentRecord[]>(client, "/ingest/documents"));
    } catch (e) {
      fail(e);
    } finally {
      setLoading(false);
    }
  }, [client, fail]);

  const remove = useCallback(
    async (id: string) => {
      try {
        await del(client, `/ingest/documents/${id}`);
        setDocs((d) => d.filter((x) => x.id !== id));
        toast("Document removed");
      } catch (e) {
        fail(e);
      }
    },
    [client, fail, toast],
  );

  const share = useCallback(
    async (id: string, payload: SharePayload) => {
      try {
        await patchJson(client, `/ingest/documents/${id}/share`, payload);
        setDocs((prev) =>
          prev.map((d) =>
            d.id === id
              ? {
                  ...d,
                  allowed_roles: payload.allowed_roles,
                  allowed_user_ids: payload.allowed_user_ids,
                }
              : d,
          ),
        );
        toast("Sharing updated");
      } catch (e) {
        fail(e);
        throw e;
      }
    },
    [client, fail, toast],
  );

  const ingestUrl = useCallback(
    async (url: string) => {
      try {
        const r = await postJson<IngestResult>(client, "/ingest", { source: url });
        toast(`Ingested ${r.chunks} chunks`);
        await refresh();
      } catch (e) {
        fail(e);
      }
    },
    [client, fail, refresh, toast],
  );

  const ingestFile = useCallback(
    async (file: File) => {
      try {
        const r = await uploadFile(client, file);
        toast(`Ingested ${r.chunks} chunks`);
        await refresh();
      } catch (e) {
        fail(e);
      }
    },
    [client, fail, refresh, toast],
  );

  return { docs, loading, refresh, remove, share, ingestUrl, ingestFile };
}
