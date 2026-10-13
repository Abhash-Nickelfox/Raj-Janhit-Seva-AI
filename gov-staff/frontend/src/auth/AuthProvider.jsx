// Who is signed in. The session lives in an httpOnly cookie; we only ask the
// API "who am I" and never keep a token in JavaScript.
import { createContext, useCallback, useContext, useEffect, useState } from "react";

import { api } from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(undefined); // undefined = still checking, null = signed out

  const refresh = useCallback(async () => {
    try {
      setUser(await api.get("/auth/me"));
    } catch {
      setUser(null);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const login = async (identifier, password) => {
    await api.post("/auth/login", { login: identifier, password });
    await refresh();
  };

  const logout = async () => {
    try {
      await api.post("/auth/logout");
    } finally {
      setUser(null);
    }
  };

  return <AuthContext.Provider value={{ user, login, logout }}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);
