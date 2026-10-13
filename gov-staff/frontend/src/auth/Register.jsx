// Staff registration: name, email, phone, department, role/designation, password.
// On success the user verifies the account by OTP (VerifyOtp.jsx).
import { useEffect, useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";

import { api } from "../api/client";
import { useT } from "../i18n";
import { useAuth } from "./AuthProvider";
import { normalizePhone, squash, validateRegistration } from "./validation";

const EMPTY = { full_name: "", email: "", phone: "", department_code: "", designation: "", password: "", confirm_password: "" };

function Field({ id, label, error, hint, children }) {
  const { t } = useT();
  return (
    <div className={`field ${error ? "has-error" : ""}`}>
      <label htmlFor={id}>{label}</label>
      {children}
      {error ? <p className="field-error" id={`${id}-err`}>{t(error)}</p> : hint ? <p className="field-hint">{hint}</p> : null}
    </div>
  );
}

export default function Register() {
  const { t, lang } = useT();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState(EMPTY);
  const [touched, setTouched] = useState({});
  const [serverErrors, setServerErrors] = useState({});
  const [formError, setFormError] = useState("");
  const [departments, setDepartments] = useState([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.get("/departments").then(setDepartments).catch(() => setFormError(t("register.noDepartments")));
  }, [t]);

  if (user) return <Navigate to="/staff" replace />;

  const errors = validateRegistration(form);
  const shown = (name) => (touched[name] && errors[name]) || serverErrors[name];
  const set = (name) => (e) => {
    setForm((f) => ({ ...f, [name]: e.target.value }));
    setServerErrors((s) => ({ ...s, [name]: undefined }));
  };
  const blur = (name) => () => setTouched((x) => ({ ...x, [name]: true }));
  const aria = (name) => ({ id: name, "aria-invalid": Boolean(shown(name)), "aria-describedby": shown(name) ? `${name}-err` : undefined });

  const submit = async (e) => {
    e.preventDefault();
    setFormError("");
    setTouched(Object.fromEntries(Object.keys(EMPTY).map((k) => [k, true])));
    if (Object.keys(errors).length) return;
    setBusy(true);
    try {
      const res = await api.post("/auth/register", {
        full_name: squash(form.full_name), email: squash(form.email).toLowerCase(), phone: normalizePhone(form.phone),
        department_code: form.department_code, designation: squash(form.designation),
        password: form.password, confirm_password: form.confirm_password,
      });
      navigate("/staff/verify", { state: { pending: res } });
    } catch (err) {
      const fields = err.detail?.fields;
      if (fields) {
        // server messages are English; show them under the matching field
        setServerErrors(Object.fromEntries(Object.entries(fields).map(([k, v]) => [k === "_form" ? "confirm_password" : k, v])));
      } else if (err.detail?.field) {
        setServerErrors({ [err.detail.field]: err.message });
      } else if (err.status === 409) {
        setFormError(t("register.duplicate"));
      } else if (err.status === 429) {
        setFormError(t("register.tooMany"));
      } else {
        setFormError(err.status === 0 ? t("login.network") : t("register.failed"));
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-wrap">
      <form className="card login register" onSubmit={submit} noValidate>
        <h1>{t("register.title")}</h1>
        <p className="sub">{t("register.subtitle")}</p>

        <Field id="full_name" label={t("register.name")} error={shown("full_name")}>
          <input {...aria("full_name")} autoComplete="name" value={form.full_name} onChange={set("full_name")} onBlur={blur("full_name")} maxLength={100} autoFocus />
        </Field>
        <Field id="email" label={t("register.email")} error={shown("email")}>
          <input {...aria("email")} type="email" autoComplete="email" value={form.email} onChange={set("email")} onBlur={blur("email")} maxLength={254} />
        </Field>
        <Field id="phone" label={t("register.phone")} error={shown("phone")} hint={t("register.phoneHint")}>
          <input {...aria("phone")} type="tel" inputMode="tel" autoComplete="tel" value={form.phone} onChange={set("phone")} onBlur={blur("phone")} maxLength={16} />
        </Field>
        <Field id="department_code" label={t("register.department")} error={shown("department_code")}>
          <select {...aria("department_code")} value={form.department_code} onChange={set("department_code")} onBlur={blur("department_code")}>
            <option value="">{t("register.chooseDepartment")}</option>
            {departments.map((d) => <option key={d.code} value={d.code}>{lang === "hi" ? d.name_hi : d.name_en}</option>)}
          </select>
        </Field>
        <Field id="designation" label={t("register.designation")} error={shown("designation")}>
          <input {...aria("designation")} autoComplete="organization-title" value={form.designation} onChange={set("designation")} onBlur={blur("designation")} maxLength={100} />
        </Field>
        <Field id="password" label={t("register.password")} error={shown("password")} hint={t("register.passwordHint")}>
          <input {...aria("password")} type="password" autoComplete="new-password" value={form.password} onChange={set("password")} onBlur={blur("password")} maxLength={64} />
        </Field>
        <Field id="confirm_password" label={t("register.confirm")} error={shown("confirm_password")}>
          <input {...aria("confirm_password")} type="password" autoComplete="new-password" value={form.confirm_password} onChange={set("confirm_password")} onBlur={blur("confirm_password")} maxLength={64} />
        </Field>

        {formError && <p className="error" role="alert">{formError}</p>}
        <button type="submit" className="primary" disabled={busy}>{busy ? t("register.registering") : t("register.submit")}</button>
        <p className="switch">{t("register.haveAccount")} <Link to="/staff/login">{t("login.submit")}</Link></p>
      </form>
    </div>
  );
}
