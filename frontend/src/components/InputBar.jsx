import { useState } from "react";
import MicButton from "./MicButton.jsx";

export default function InputBar({ onSend, disabled, lastReplyLang }) {
  const [text, setText] = useState("");
  const [voiceError, setVoiceError] = useState("");

  const submit = () => {
    if (!text.trim() || disabled) return;
    onSend(text);
    setText("");
  };

  return (
    <div style={{ borderTop: "1px solid #e5e7eb" }}>
      {voiceError && (
        <div style={{ padding: "6px 14px 0", fontSize: 12, color: "#dc2626" }}>{voiceError}</div>
      )}
      <div style={{ display: "flex", gap: 8, padding: 10 }}>
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          disabled={disabled}
          placeholder="Type / likhe / लिखें..."
          style={{
            flex: 1,
            padding: "10px 12px",
            borderRadius: 20,
            border: "1px solid #d1d5db",
            fontSize: 14,
            outline: "none",
          }}
        />
        <MicButton
          onSend={(voiceText) => {
            setVoiceError("");
            onSend(voiceText);
          }}
          disabled={disabled}
          onError={setVoiceError}
          lastReplyLang={lastReplyLang}
        />
        <button
          onClick={submit}
          disabled={disabled}
          aria-label="Send"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 40,
            height: 40,
            padding: 0,
            borderRadius: "50%",
            border: "none",
            background: disabled ? "#9ca3af" : "#1a56db",
            color: "#fff",
            cursor: disabled ? "default" : "pointer",
            flexShrink: 0,
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 20L21 12L3 4V10L16 12L3 14V20Z" fill="currentColor" />
          </svg>
        </button>
      </div>
    </div>
  );
}
