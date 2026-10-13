// Common start screen: choose Citizen or Government Staff. Same look as the
// citizen chat (card, brand header, side illustrations).
import { useNavigate } from "react-router-dom";

import logoFull from "../assets/logo-full.png";
import sideLeft from "../assets/figma/side-illustration-left.png";
import sideRight from "../assets/figma/side-illustration-right.png";
import { useT } from "../i18n";
import "../styles/chat.css";

export default function ChooseUserType() {
  const { t, lang, setLang } = useT();
  const navigate = useNavigate();

  return (
    <div className="chat-page">
      <img src={sideLeft} alt="" className="side side-left" />
      <img src={sideRight} alt="" className="side side-right" />
      <div className="chat-column">
        <div className="chat-card">
          <div className="chat-header">
            <div className="brand-logo"><img src={logoFull} alt="Raj Janhit Seva AI" height={64} /></div>
            <div className="brand-text">
              <p className="brand-name">राज जनहित सेवा<span>.ai</span></p>
              <p className="brand-sub">{t("choose.tagline")}</p>
            </div>
            <button type="button" className="pill-btn" onClick={() => setLang(lang === "en" ? "hi" : "en")}>{t("app.language")}</button>
          </div>

          <div className="choose-body">
            <p className="choose-hi">नमस्ते! आप कौन हैं?</p>
            <p className="choose-en">{t("choose.title")}</p>
            <div className="choose-grid">
              <button type="button" className="choose-card" onClick={() => navigate("/citizen")}>
                <span className="choose-icon" aria-hidden="true">🧑‍🌾</span>
                <b>{t("choose.citizen")}</b>
                <span>{t("choose.citizenText")}</span>
                <span className="choose-go">{t("choose.citizenGo")} →</span>
              </button>
              <button type="button" className="choose-card" onClick={() => navigate("/staff")}>
                <span className="choose-icon" aria-hidden="true">🏛️</span>
                <b>{t("choose.staff")}</b>
                <span>{t("choose.staffText")}</span>
                <span className="choose-go">{t("choose.staffGo")} →</span>
              </button>
            </div>
          </div>
        </div>
        <p className="chat-foot">{t("choose.helpline")}</p>
      </div>
    </div>
  );
}
