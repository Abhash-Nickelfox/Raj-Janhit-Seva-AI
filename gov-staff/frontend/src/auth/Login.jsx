// Sign in with the registered email or phone number + password.
import { useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";

import { useT } from "../i18n";
import { useAuth } from "./AuthProvider";
import { normalizePhone, squash } from "./validation";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

export default function Login() {
  const { t } = useT();
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [ident, setIdent] = useState(location.state?.login || "");
  const [password, setPassword] = useState("");
  const [touched, setTouched] = useState({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to={location.state?.from || "/staff"} replace />;

  const value = squash(ident);
  const identError = !value ? "register.err.required"
    : value.includes("@") ? (EMAIL_RE.test(value) ? null : "register.err.email")
      : normalizePhone(value) ? null : "login.err.identifier";
  const passwordError = password ? null : "register.err.required";

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setTouched({ ident: true, password: true });
    if (identError || passwordError) return;
    setBusy(true);
    try {
      await login(value.includes("@") ? value.toLowerCase() : normalizePhone(value), password);
      navigate(location.state?.from || "/staff", { replace: true });
    } catch (err) {
      if (err.status === 403 && err.detail?.code === "not_verified") {
        navigate("/staff/verify", { state: { pending: err.detail } });
        return;
      }
      setError(err.status === 429 ? t("login.tooMany") : err.status === 0 ? t("login.network") : t("login.bad"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-wrap">
      <form className="card login" onSubmit={submit} noValidate>
        <h1>{t("login.title")}</h1>
        <p className="sub">{t("login.subtitle")}</p>
        <div className={`field ${touched.ident && identError ? "has-error" : ""}`}>
          <label htmlFor="ident">{t("login.identifier")}</label>
          <input id="ident" autoComplete="username" value={ident} onChange={(e) => setIdent(e.target.value)}
            onBlur={() => setTouched((x) => ({ ...x, ident: true }))} aria-invalid={Boolean(touched.ident && identError)}
            maxLength={254} autoFocus={!ident} />
          {touched.ident && identError && <p className="field-error">{t(identError)}</p>}
        </div>
        <div className={`field ${touched.password && passwordError ? "has-error" : ""}`}>
          <label htmlFor="password">{t("login.password")}</label>
          <input id="password" type="password" autoComplete="current-password" value={password}
            onChange={(e) => setPassword(e.target.value)} onBlur={() => setTouched((x) => ({ ...x, password: true }))}
            aria-invalid={Boolean(touched.password && passwordError)} maxLength={64} autoFocus={Boolean(ident)} />
          {touched.password && passwordError && <p className="field-error">{t(passwordError)}</p>}
        </div>
        {error && <p className="error" role="alert">{error}</p>}
        <button type="submit" className="primary" disabled={busy}>{busy ? t("login.signingIn") : t("login.submit")}</button>
        <p className="switch">{t("login.newHere")} <Link to="/staff/register">{t("register.submit")}</Link></p>
        <p className="note">{t("login.isolation")}</p>
      </form>
    </div>
  );
}
