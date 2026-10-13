// Verify a new registration: choose email or phone -> send OTP -> enter it ->
// "Registration successful" popup -> sign in. (Testing: the code is always 1234.)
import { useEffect, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";

import { api } from "../api/client";
import { useT } from "../i18n";

export default function VerifyOtp() {
  const { t } = useT();
  const navigate = useNavigate();
  const pending = useLocation().state?.pending; // {email, email_masked, phone_masked, has_phone}
  const [channel, setChannel] = useState("email");
  const [sent, setSent] = useState(null); // {sent_to, channel, test_mode}
  const [code, setCode] = useState("");
  const [touched, setTouched] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [wait, setWait] = useState(0);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (wait <= 0) return undefined;
    const id = setTimeout(() => setWait((w) => w - 1), 1000);
    return () => clearTimeout(id);
  }, [wait]);

  if (!pending) return <Navigate to="/staff/register" replace />;

  const codeError = !code ? "register.err.required" : /^\d{4,6}$/.test(code) ? null : "verify.err.code";

  const send = async () => {
    setError("");
    setBusy(true);
    try {
      const res = await api.post("/auth/register/send-otp", { email: pending.email, channel });
      setSent(res);
      setCode("");
      setTouched(false);
      setWait(res.resend_after || 60);
    } catch (err) {
      if (err.detail?.retry_after) setWait(err.detail.retry_after);
      setError(err.status === 0 ? t("login.network") : err.message);
    } finally {
      setBusy(false);
    }
  };

  const verify = async (e) => {
    e.preventDefault();
    setTouched(true);
    setError("");
    if (codeError) return;
    setBusy(true);
    try {
      await api.post("/auth/register/verify-otp", { email: pending.email, code });
      setDone(true);
    } catch (err) {
      setError(err.status === 0 ? t("login.network") : err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-wrap">
      <div className="card login">
        <h1>{t("verify.title")}</h1>
        <p className="sub">{t("verify.subtitle")}</p>

        {!sent ? (
          <>
            <fieldset className="choice">
              <legend>{t("verify.chooseMethod")}</legend>
              <label className={`option ${channel === "email" ? "on" : ""}`}>
                <input type="radio" name="channel" value="email" checked={channel === "email"} onChange={() => setChannel("email")} />
                <span><b>{t("verify.byEmail")}</b><small>{pending.email_masked}</small></span>
              </label>
              {pending.has_phone && (
                <label className={`option ${channel === "phone" ? "on" : ""}`}>
                  <input type="radio" name="channel" value="phone" checked={channel === "phone"} onChange={() => setChannel("phone")} />
                  <span><b>{t("verify.byPhone")}</b><small>{pending.phone_masked}</small></span>
                </label>
              )}
            </fieldset>
            {error && <p className="error" role="alert">{error}</p>}
            <button type="button" className="primary" onClick={send} disabled={busy || wait > 0}>
              {busy ? t("verify.sending") : wait > 0 ? t("verify.waitSeconds", { s: wait }) : t("verify.send")}
            </button>
          </>
        ) : (
          <form onSubmit={verify} noValidate className="stack-form">
            <p className="sent">{t(sent.channel === "phone" ? "verify.sentPhone" : "verify.sentEmail", { to: sent.sent_to })}</p>
            {sent.test_mode && <p className="note">{t("verify.testMode")}</p>}
            <div className={`field ${touched && codeError ? "has-error" : ""}`}>
              <label htmlFor="otp">{t("verify.code")}</label>
              <input id="otp" className="otp" inputMode="numeric" autoComplete="one-time-code" maxLength={6} autoFocus
                value={code} onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))} onBlur={() => setTouched(true)}
                aria-invalid={Boolean(touched && codeError)} />
              {touched && codeError && <p className="field-error">{t(codeError)}</p>}
            </div>
            {error && <p className="error" role="alert">{error}</p>}
            <button type="submit" className="primary" disabled={busy}>{busy ? t("verify.verifying") : t("verify.verify")}</button>
            <div className="otp-actions">
              <button type="button" className="link" onClick={send} disabled={busy || wait > 0}>
                {wait > 0 ? t("verify.resendIn", { s: wait }) : t("verify.resend")}
              </button>
              <button type="button" className="link" onClick={() => { setSent(null); setError(""); }}>{t("verify.changeMethod")}</button>
            </div>
          </form>
        )}
      </div>

      {done && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="reg-ok-title">
          <div className="modal">
            <div className="modal-icon" aria-hidden="true">✓</div>
            <h2 id="reg-ok-title">{t("register.successTitle")}</h2>
            <p>{t("register.successText")}</p>
            <button type="button" className="primary" autoFocus
              onClick={() => navigate("/staff/login", { replace: true, state: { login: pending.email } })}>
              {t("register.goToSignIn")}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
