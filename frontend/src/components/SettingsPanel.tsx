import { motion } from "framer-motion";
import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { useSettings } from "../context/SettingsContext";
import { ApiError } from "../api/client";

export function SettingsPanel({ onClose }: { onClose: () => void }) {
  const { client, setBaseUrl, setApiKey } = useSettings();
  const { user, login, logout } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onLogin() {
    setBusy(true);
    setError("");
    try {
      await login(username, password);
      setPassword("");
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="absolute bottom-full left-0 z-50 mb-2 w-80 rounded-lg border border-line bg-surface p-4 shadow-xl"
    >
      <label className="block text-xs font-medium text-muted">API URL</label>
      <input
        className="mt-1 w-full rounded border border-muted/50 px-2 py-1 text-sm"
        value={client.baseUrl}
        onChange={(e) => setBaseUrl(e.target.value)}
      />

      <div className="mt-3 border-t border-line pt-3">
        <p className="text-xs font-medium text-muted">Account</p>
        {user ? (
          <div className="mt-2 space-y-2">
            <p className="text-sm text-ink">
              Signed in as <span className="font-medium">{user.username}</span>
              <span className="ml-1 text-xs text-muted">({user.roles.join(", ")})</span>
            </p>
            <button
              type="button"
              className="w-full rounded border border-line px-3 py-1.5 text-sm hover:bg-sunken"
              onClick={logout}
            >
              Sign out
            </button>
          </div>
        ) : (
          <div className="mt-2 space-y-2">
            <input
              className="w-full rounded border border-muted/50 px-2 py-1 text-sm"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
            />
            <input
              type="password"
              className="w-full rounded border border-muted/50 px-2 py-1 text-sm"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
            />
            {error && <p className="text-xs text-danger">{error}</p>}
            <button
              type="button"
              disabled={busy || !username || !password}
              className="w-full rounded bg-primary px-3 py-1.5 text-sm text-white hover:bg-primary-hover disabled:opacity-50"
              onClick={() => void onLogin()}
            >
              {busy ? "Signing in…" : "Sign in"}
            </button>
          </div>
        )}
      </div>

      <label className="mt-3 block text-xs font-medium text-muted">
        Bearer token (JWT or API key)
      </label>
      <input
        type="password"
        className="mt-1 w-full rounded border border-muted/50 px-2 py-1 text-sm"
        value={client.apiKey ?? ""}
        onChange={(e) => setApiKey(e.target.value)}
      />
      <button
        className="mt-4 w-full rounded bg-primary px-3 py-1.5 text-sm text-white hover:bg-primary-hover"
        onClick={onClose}
      >
        Done
      </button>
    </motion.div>
  );
}
