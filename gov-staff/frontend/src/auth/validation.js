// Same rules as the backend (app/auth/validation.py), for instant feedback.
// The server re-checks everything; these only return i18n keys.
const NAME_RE = /^[A-Za-zऀ-ॿ][A-Za-zऀ-ॿ .'-]{1,99}$/;
const DESIGNATION_RE = /^[A-Za-z0-9ऀ-ॿ][A-Za-z0-9ऀ-ॿ .,&()/'-]{1,99}$/;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

export const squash = (v) => (v || "").replace(/\s+/g, " ").trim();

export function normalizePhone(value) {
  let d = (value || "").replace(/[\s\-()]/g, "");
  if (d.startsWith("+91")) d = d.slice(3);
  else if (d.length === 12 && d.startsWith("91")) d = d.slice(2);
  else if (d.length === 11 && d.startsWith("0")) d = d.slice(1);
  return /^[6-9]\d{9}$/.test(d) ? d : null;
}

export function validateRegistration(f) {
  const e = {};
  if (!squash(f.full_name)) e.full_name = "register.err.required";
  else if (!NAME_RE.test(squash(f.full_name))) e.full_name = "register.err.name";

  if (!squash(f.email)) e.email = "register.err.required";
  else if (!EMAIL_RE.test(squash(f.email)) || squash(f.email).length > 254) e.email = "register.err.email";

  if (!squash(f.phone)) e.phone = "register.err.required";
  else if (!normalizePhone(f.phone)) e.phone = "register.err.phone";

  if (!f.department_code) e.department_code = "register.err.department";

  if (!squash(f.designation)) e.designation = "register.err.required";
  else if (!DESIGNATION_RE.test(squash(f.designation))) e.designation = "register.err.designation";

  const pw = f.password || "";
  if (!pw) e.password = "register.err.required";
  else if (pw.length < 8 || pw.length > 64) e.password = "register.err.passwordLength";
  else if (pw !== pw.trim()) e.password = "register.err.passwordSpace";
  else if (!/[A-Za-z]/.test(pw) || !/\d/.test(pw)) e.password = "register.err.passwordMix";

  if (!f.confirm_password) e.confirm_password = "register.err.required";
  else if (f.confirm_password !== pw) e.confirm_password = "register.err.mismatch";
  return e;
}
