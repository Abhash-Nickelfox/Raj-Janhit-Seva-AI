// Same design as the citizen SuggestionsPanel, with questions staff ask about
// their department's documents.
import { useState } from "react";

import iconSuggested from "../../assets/figma/icon-suggested.svg";
import { useT } from "../../i18n";

const TABS = ["suggested", "circulars", "schemes", "procedures"];

export default function StaffSuggestions({ onSend }) {
  const { t } = useT();
  const [active, setActive] = useState("suggested");
  const prompts = [1, 2, 3, 4].map((n) => t(`staffChat.prompts.${active}.${n}`));

  return (
    <div style={{ display: "flex", flexDirection: "column", minWidth: 0, gap: 4 }}>
      <div style={{ display: "flex", gap: 8, paddingBottom: 8, paddingLeft: 2, paddingRight: 2, borderBottom: "1px solid var(--color-tab-border)", overflowX: "auto" }}>
        {TABS.map((tab) => {
          const on = tab === active;
          return (
            <button key={tab} onClick={() => setActive(tab)}
              style={{ display: "flex", alignItems: "center", gap: 8, minHeight: 42, padding: "0 15px", borderRadius: 999, border: "none", background: on ? "var(--color-tab-active-bg)" : "transparent", cursor: "pointer", flexShrink: 0 }}>
              {tab === "suggested" && <img src={iconSuggested} alt="" width={15} height={15} />}
              <span style={{ fontFamily: "var(--font-latin), var(--font-devanagari)", fontWeight: on ? 600 : 500, fontSize: 14.4, color: on ? "var(--color-ink)" : "var(--color-ink-faint)", whiteSpace: "nowrap" }}>
                {t(`staffChat.tabs.${tab}`)}
              </span>
            </button>
          );
        })}
      </div>
      <div style={{ display: "flex", flexDirection: "column" }}>
        {prompts.map((text, i) => (
          <button key={text} onClick={() => onSend(text)}
            style={{ display: "flex", alignItems: "center", gap: 15, minHeight: 60, padding: 16, border: "none", borderBottom: i < prompts.length - 1 ? "1px solid var(--color-tab-border)" : "none", background: "transparent", cursor: "pointer", textAlign: "left", width: "100%" }}>
            <span aria-hidden="true" style={{ fontSize: 17, flexShrink: 0 }}>📄</span>
            <span style={{ fontFamily: "var(--font-latin), var(--font-devanagari)", fontSize: 16.16, lineHeight: "22.624px", color: "var(--color-ink)" }}>{text}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
