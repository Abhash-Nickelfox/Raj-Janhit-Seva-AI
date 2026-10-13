import { Navigate, useLocation } from "react-router-dom";

import { useT } from "../i18n";
import { useAuth } from "./AuthProvider";

export default function RequireAuth({ children }) {
  const { user } = useAuth();
  const { t } = useT();
  const location = useLocation();
  if (user === undefined) return <p className="loading">{t("common.loading")}</p>;
  if (!user) return <Navigate to="/staff/login" replace state={{ from: location.pathname }} />;
  return children;
}
