// Frame for the staff sign-in pages (login, register, verify). Brand header
// matches the citizen app; the chat page has its own header (StaffChatPage).
import { Link, Outlet } from "react-router-dom";

import logoFull from "../assets/logo-full.png";
import { useT } from "../i18n";

export default function StaffShell() {
  const { t, lang, setLang } = useT();
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand" aria-label={t("choose.change")}>
          <div className="brand-logo">
            <img src={logoFull} alt="Raj Janhit Seva AI" height={64} />
          </div>
          <div className="brand-text">
            <p className="brand-name">
              राज जनहित सेवा<span>.ai</span>
            </p>
            <p className="brand-sub">{t("app.panel")}</p>
          </div>
        </Link>
        <div className="top-actions">
          <Link to="/" className="lang">← {t("choose.change")}</Link>
          <button type="button" className="lang" onClick={() => setLang(lang === "en" ? "hi" : "en")}>
            {t("app.language")}
          </button>
        </div>
      </header>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
