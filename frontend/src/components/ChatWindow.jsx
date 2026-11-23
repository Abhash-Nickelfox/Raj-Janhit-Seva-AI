import { useEffect, useRef, useState } from "react";
import { useChatStore } from "../store/chatStore.js";
import Header from "./Header.jsx";
import EmptyState from "./EmptyState.jsx";
import SuggestionsPanel from "./SuggestionsPanel.jsx";
import MessageBubble from "./MessageBubble.jsx";
import InputBar from "./InputBar.jsx";
import SmsPopup from "./SmsPopup.jsx";

export default function ChatWindow() {
  const {
    messages,
    isLoading,
    sendMessage,
    smsPopup,
    dismissSmsPopup,
    preferredLang,
    setPreferredLang,
  } = useChatStore();
  const bottomRef = useRef(null);
  const [ideasOn, setIdeasOn] = useState(true);

  const hasConversationStarted = messages.some((m) => m.role === "user");

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
          position: "relative",
          maxHeight: hasConversationStarted ? "calc(100vh - 78px)" : "none",
          background: "#ffffff",
          border: "1px solid var(--color-card-border)",
          borderRadius: 24,
          boxShadow: "0px 1px 2px 0px rgba(17,26,46,0.04), 0px 18px 44px -30px rgba(17,26,46,0.4)",
          overflow: "hidden",
        }}
      >
        <SmsPopup text={smsPopup} onDismiss={dismissSmsPopup} />
        <Header />

        {!hasConversationStarted && <EmptyState />}

        {hasConversationStarted && (
          <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
            {messages.map((m, i) => (
              <MessageBubble key={i} role={m.role} text={m.text} />
            ))}
            {isLoading && (
              <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 10 }}>
                <div
                  style={{
                    padding: "10px 14px",
                    borderRadius: 14,
                    background: "#ffffff",
                    border: "1px solid var(--color-card-border)",
                    color: "var(--color-ink-faint)",
                    fontSize: 14,
                  }}
                >
                  ...
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}

        <InputBar
          onSend={sendMessage}
          disabled={isLoading}
          preferredLang={preferredLang}
          onChangeLang={setPreferredLang}
          ideasOn={ideasOn}
          onToggleIdeas={() => setIdeasOn((v) => !v)}
        />
      </div>

      {ideasOn && <SuggestionsPanel onSend={sendMessage} />}

      {!hasConversationStarted && (
        <p
          style={{
            margin: 0,
            textAlign: "center",
            fontFamily: "var(--font-latin)",
            fontWeight: 400,
            fontSize: 13.6,
            lineHeight: "22.032px",
            letterSpacing: -0.051,
            color: "var(--color-ink-faint)",
          }}
        >
          Can't read or type? Call 181 - toll-free, available 24×7.
        </p>
      )}
    </div>
  );
}
