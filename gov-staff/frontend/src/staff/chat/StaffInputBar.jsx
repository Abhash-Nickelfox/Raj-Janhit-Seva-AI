// Same design as the citizen InputBar. Voice input is left out for staff for now
// (the citizen app's speech service belongs to the citizen backend).
import { useState } from "react";

import iconSend from "../../assets/figma/icon-send.svg";
import iconSparkle from "../../assets/figma/icon-sparkle.svg";
import { useT } from "../../i18n";
import LanguageSelector from "./LanguageSelector";

export default function StaffInputBar({ onSend, disabled, preferredLang, onChangeLang, ideasOn, onToggleIdeas }) {
  const { t } = useT();
  const [text, setText] = useState("");

  const submit = () => {
    if (!text.trim() || disabled) return;
    onSend(text);
    setText("");
  };

  return (
    <div style={{ padding: "12px 14px 14px" }}>
      <div style={{ background: "var(--color-bg)", border: "1px solid var(--color-card-border)", borderRadius: 18, padding: "6px 6px 8px", display: "flex", flexDirection: "column", minWidth: 0, gap: 2 }}>
        <div style={{ padding: "16px 16px 10px" }}>
          <input
            type="text"
            value={text}
            maxLength={2000}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
            disabled={disabled}
            placeholder={t("staffChat.placeholder")}
            aria-label={t("staffChat.placeholder")}
            style={{ width: "100%", border: "none", outline: "none", background: "transparent", fontFamily: "var(--font-latin), var(--font-devanagari)", fontSize: 16.8, color: "var(--color-ink)" }}
          />
        </div>
        <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 9, padding: "2px 6px 2px 8px" }}>
          <LanguageSelector value={preferredLang} onChange={onChangeLang} disabled={disabled} />
          <button
            onClick={onToggleIdeas}
            style={{ display: "flex", alignItems: "center", gap: 7, minHeight: 40, padding: "0 15px", borderRadius: 999, border: ideasOn ? "1px solid transparent" : "1px solid var(--color-card-border)", background: ideasOn ? "var(--color-accent-blue-bg)" : "transparent", cursor: "pointer", flexShrink: 0 }}
          >
            <img src={iconSparkle} alt="" width={14} height={14} />
            <span style={{ fontFamily: "var(--font-latin)", fontWeight: 600, fontSize: 14.4, color: ideasOn ? "var(--color-accent-blue)" : "var(--color-ink-muted)", whiteSpace: "nowrap" }}>
              {t("staffChat.ideas")}
            </span>
          </button>
          <button
            onClick={submit}
            disabled={disabled}
            aria-label={t("staffChat.send")}
            style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 44, height: 44, padding: 0, marginLeft: "auto", borderRadius: 999, border: "none", background: disabled ? "#9ca3af" : "var(--color-send-bg)", cursor: disabled ? "default" : "pointer", flexShrink: 0 }}
          >
            <img src={iconSend} alt="" width={17} height={17} style={{ display: "block" }} />
          </button>
        </div>
      </div>
    </div>
  );
}
