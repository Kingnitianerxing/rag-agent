import type { IngestResult } from "./types";

export class ApiError extends Error {
  constructor(public status: number, public detail: string) {
    super(detail);
    this.name = "ApiError";
  }
}

export interface ClientOptions {
  baseUrl: string;
  apiKey?: string;
}

function authHeaders(o: ClientOptions, extra: Record<string, string> = {}): Record<string, string> {
  return o.apiKey ? { ...extra, Authorization: `Bearer ${o.apiKey}` } : { ...extra };
}

async function detailOf(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    if (data?.detail) return JSON.stringify(data.detail);
    return JSON.stringify(data);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

export async function getJson<T>(o: ClientOptions, path: string): Promise<T> {
  const res = await fetch(o.baseUrl + path, { headers: authHeaders(o) });
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  return (await res.json()) as T;
}

export async function postJson<T>(o: ClientOptions, path: string, body: unknown): Promise<T> {
  const res = await fetch(o.baseUrl + path, {
    method: "POST",
    headers: authHeaders(o, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  return (await res.json()) as T;
}

export async function patchJson<T>(o: ClientOptions, path: string, body: unknown): Promise<T> {
  const res = await fetch(o.baseUrl + path, {
    method: "PATCH",
    headers: authHeaders(o, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  return (await res.json()) as T;
}

export async function del<T>(o: ClientOptions, path: string): Promise<T> {
  const res = await fetch(o.baseUrl + path, { method: "DELETE", headers: authHeaders(o) });
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  return (await res.json()) as T;
}

export async function postStream(o: ClientOptions, path: string, body: unknown): Promise<ReadableStream<Uint8Array>> {
  const res = await fetch(o.baseUrl + path, {
    method: "POST",
    headers: authHeaders(o, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  if (!res.body) throw new ApiError(0, "empty response body");
  return res.body;
}

export async function uploadFile(o: ClientOptions, file: File): Promise<IngestResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(o.baseUrl + "/ingest/upload", { method: "POST", headers: authHeaders(o), body: form });
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  return (await res.json()) as IngestResult;
}
