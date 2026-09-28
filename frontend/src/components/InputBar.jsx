import { useState } from "react";
import MicButton from "./MicButton.jsx";
import LanguageSelector from "./LanguageSelector.jsx";
import iconSparkle from "../assets/figma/icon-sparkle.svg";
import iconSend from "../assets/figma/icon-send.svg";

export default function InputBar({ onSend, disabled, preferredLang, onChangeLang, ideasOn, onToggleIdeas }) {
  const [text, setText] = useState("");
  const [voiceError, setVoiceError] = useState("");

  const submit = () => {
    if (!text.trim() || disabled) return;
    onSend(text);
    setText("");
  };

  return (
    <div style={{ padding: "12px 14px 14px" }}>
      {voiceError && (
        <div style={{ padding: "0 6px 6px", fontSize: 12, color: "#dc2626", fontFamily: "var(--font-latin)" }}>
          {voiceError}
        </div>
      )}
      <div
        style={{
          background: "var(--color-bg)",
          border: "1px solid var(--color-card-border)",
          borderRadius: 18,
          padding: "6px 6px 8px",
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
          gap: 2,
        }}
      >
        <div style={{ padding: "16px 16px 10px" }}>
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
            disabled={disabled}
            placeholder="Ask about a scheme, job or benefit…"
            style={{
              width: "100%",
              border: "none",
              outline: "none",
              background: "transparent",
              fontFamily: "var(--font-latin)",
              fontWeight: 400,
              fontSize: 16.8,
              color: "var(--color-ink)",
            }}
          />
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 9,
            padding: "2px 6px 2px 8px",
          }}
        >
          <MicButton
            onSend={(voiceText) => {
              setVoiceError("");
              onSend(voiceText);
            }}
            disabled={disabled}
            onError={setVoiceError}
          />

          <LanguageSelector value={preferredLang} onChange={onChangeLang} disabled={disabled} />

          <button
            onClick={onToggleIdeas}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 7,
              minHeight: 40,
              padding: "0 15px",
              borderRadius: 999,
              border: ideasOn ? "1px solid transparent" : "1px solid var(--color-card-border)",
              background: ideasOn ? "var(--color-accent-blue-bg)" : "transparent",
              cursor: "pointer",
              flexShrink: 0,
            }}
          >
            <img src={iconSparkle} alt="" width={14} height={14} />
            <span
              style={{
                fontFamily: "var(--font-latin)",
                fontWeight: 600,
                fontSize: 14.4,
                color: ideasOn ? "var(--color-accent-blue)" : "var(--color-ink-muted)",
                whiteSpace: "nowrap",
              }}
            >
              Ideas
            </span>
          </button>

          <button
            onClick={submit}
            disabled={disabled}
            aria-label="Send"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 44,
              height: 44,
              padding: 0,
              marginLeft: "auto",
              borderRadius: 999,
              border: "none",
              background: disabled ? "#9ca3af" : "var(--color-send-bg)",
              cursor: disabled ? "default" : "pointer",
              flexShrink: 0,
            }}
          >
            <img src={iconSend} alt="" width={17} height={17} style={{ display: "block" }} />
          </button>
        </div>
      </div>
    </div>
  );
}
