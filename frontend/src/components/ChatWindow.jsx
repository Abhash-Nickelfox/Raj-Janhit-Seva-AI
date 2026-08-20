import { useEffect, useRef } from "react";
import { useChatStore } from "../store/chatStore.js";
import MessageBubble from "./MessageBubble.jsx";
import InputBar from "./InputBar.jsx";
import SmsPopup from "./SmsPopup.jsx";

export default function ChatWindow() {
  const { messages, isLoading, sendMessage, retryLast, smsPopup, dismissSmsPopup, notice, clearNotice } =
    useChatStore();
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", position: "relative" }}>
      <SmsPopup text={smsPopup} onDismiss={dismissSmsPopup} />
      <div style={{ background: "#ff9933", color: "#fff", padding: "14px 16px" }}>
        <div style={{ fontSize: 18, fontWeight: 700 }}>जन सेवा AI 🏛️</div>
        <div style={{ fontSize: 12, opacity: 0.9 }}>Rajasthan Government Scheme Assistant</div>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: 14, background: "#f9fafb" }}>
        {messages.map((m, i) => (
          <MessageBubble
            key={i}
            role={m.role}
            text={m.text}
            onRetry={m.retryable && i === messages.length - 1 && !isLoading ? retryLast : undefined}
          />
        ))}
        {isLoading && (
          <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 10 }}>
            <div
              style={{
                padding: "10px 14px",
                borderRadius: 14,
                background: "#ffffff",
                border: "1px solid #d1d5db",
                color: "#6b7280",
                fontSize: 14,
              }}
            >
              ...
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {notice && (
        <div
          role="status"
          style={{ padding: "6px 14px", fontSize: 12, color: "#92400e", background: "#fef3c7", display: "flex", gap: 8 }}
        >
          <span style={{ flex: 1 }}>{notice}</span>
          <button onClick={clearNotice} aria-label="Dismiss" style={{ border: "none", background: "none", cursor: "pointer" }}>
            x
          </button>
        </div>
      )}
      <InputBar onSend={sendMessage} disabled={isLoading} />
    </div>
  );
}
