// Government Staff AI Assistant - same layout and design as the citizen chat,
// connected to the staff backend. The header shows who is signed in and their
// department, so staff always know whose documents they are asking about.
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import logoFull from "../../assets/logo-full.png";
import sideLeft from "../../assets/figma/side-illustration-left.png";
import sideRight from "../../assets/figma/side-illustration-right.png";
import { useAuth } from "../../auth/AuthProvider";
import ChangeUserType from "../../components/ChangeUserType";
import { useT } from "../../i18n";
import "../../styles/chat.css";
import MessageBubble from "./MessageBubble";
import StaffInputBar from "./StaffInputBar";
import StaffSuggestions from "./StaffSuggestions";
import { useStaffChatStore } from "./staffChatStore";

export default function StaffChatPage() {
  const { t, lang, setLang } = useT();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const { messages, isLoading, sendMessage, preferredLang, setPreferredLang, reset } = useStaffChatStore();
  const [ideasOn, setIdeasOn] = useState(true);
  const bottomRef = useRef(null);
  const started = messages.some((m) => m.role === "user");
  const dept = user.department;
  const deptName = dept ? (lang === "hi" ? dept.name_hi : dept.name_en) : "";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const signOut = async () => {
    await logout();
    reset();
    navigate("/staff/login", { replace: true });
  };

  return (
    <div className="chat-page">
      <ChangeUserType />
      <img src={sideLeft} alt="" className="side side-left" />
      <img src={sideRight} alt="" className="side side-right" />
      <div className="chat-column">
        <div className={`chat-card ${started ? "started" : ""}`}>
          <div className="chat-header">
            <div className="brand-logo"><img src={logoFull} alt="Raj Janhit Seva AI" height={64} /></div>
            <div className="brand-text">
              <p className="brand-name">राज जनहित सेवा<span>.ai</span></p>
              <p className="brand-sub">{t("app.panel")}{deptName ? ` · ${deptName}` : ""}</p>
            </div>
            <div className="header-right">
              <span className="online">{t("staffChat.online")}</span>
              <button type="button" className="pill-btn" onClick={() => setLang(lang === "en" ? "hi" : "en")}>{t("app.language")}</button>
              <button type="button" className="pill-btn" onClick={signOut} title={user.email}>{t("login.signOut")}</button>
            </div>
          </div>

          {!started && (
            <div style={{ padding: "28px 24px 20px" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 4, paddingTop: 12, maxWidth: 570 }}>
                <p style={{ margin: 0, fontFamily: "var(--font-devanagari)", fontWeight: 700, fontSize: 17.5, lineHeight: "27.936px", color: "var(--color-ink)" }}>
                  नमस्ते {user.full_name}! मैं {dept?.name_hi || "आपके विभाग"} के दस्तावेज़ों से जुड़े प्रश्नों में मदद करता हूं।
                </p>
                <p style={{ margin: 0, fontSize: 15.52, lineHeight: "25.142px", letterSpacing: -0.051, color: "var(--color-ink-muted)" }}>
                  Hi {user.full_name}! I help with questions about the {dept?.name_en || "your department"} department's documents.
                </p>
                <p style={{ margin: "8px 0 0", fontSize: 13.5, color: "var(--color-ink-faint)" }}>{t("staffChat.isolation", { department: deptName })}</p>
              </div>
            </div>
          )}

          {started && (
            <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
              {messages.map((m, i) => <MessageBubble key={i} role={m.role} text={m.text} />)}
              {isLoading && (
                <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 10 }}>
                  <div style={{ padding: "10px 14px", borderRadius: 14, background: "#fff", border: "1px solid var(--color-card-border)", color: "var(--color-ink-faint)", fontSize: 14 }}>...</div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          )}

          <StaffInputBar onSend={sendMessage} disabled={isLoading} preferredLang={preferredLang} onChangeLang={setPreferredLang}
            ideasOn={ideasOn} onToggleIdeas={() => setIdeasOn((v) => !v)} />
        </div>

        {ideasOn && <StaffSuggestions onSend={sendMessage} />}
      </div>
    </div>
  );
}
