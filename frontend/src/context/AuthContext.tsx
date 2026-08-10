import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { ApiError, getJson, postJson } from "../api/client";
import { useSettings } from "./SettingsContext";

export interface AuthUser {
  id: number;
  username: string;
  roles: string[];
  is_service?: boolean;
}

interface AuthValue {
  user: AuthUser | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  refreshMe: () => Promise<void>;
  canIngest: boolean;
  isAdmin: boolean;
}

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const { client, setApiKey } = useSettings();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshMe = useCallback(async () => {
    if (!client.apiKey) {
      setUser(null);
      setLoading(false);
      return;
    }
    try {
      const me = await getJson<AuthUser>(client, "/auth/me");
      setUser(me);
    } catch (e) {
      if (e instanceof ApiError && (e.status === 401 || e.status === 503)) {
        setUser(null);
      }
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => {
    void refreshMe();
  }, [refreshMe]);

  const login = useCallback(
    async (username: string, password: string) => {
      const res = await postJson<{ access_token: string; user: AuthUser }>(
        { baseUrl: client.baseUrl },
        "/auth/login",
        { username, password },
      );
      setApiKey(res.access_token);
      setUser(res.user);
    },
    [client.baseUrl, setApiKey],
  );

  const logout = useCallback(() => {
    setApiKey("");
    setUser(null);
  }, [setApiKey]);

  const roles = user?.roles ?? [];
  const isAdmin = Boolean(user?.is_service || roles.includes("admin"));
  const canIngest = isAdmin || roles.includes("editor");

  const value = useMemo(
    () => ({ user, loading, login, logout, refreshMe, canIngest, isAdmin }),
    [user, loading, login, logout, refreshMe, canIngest, isAdmin],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const v = useContext(AuthContext);
  if (!v) throw new Error("useAuth must be used within AuthProvider");
  return v;
}
