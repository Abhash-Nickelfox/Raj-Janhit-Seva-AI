import { useEffect } from "react";

// Mock-only (2026-08-17): simulates an incoming SMS notification for the
// Jan Aadhaar OTP, since in mock mode our own backend generates the OTP and
// would otherwise only ever print it to server logs - a real citizen would
// see this arrive on their actual phone instead. Auto-dismisses on its own
// (like a real notification banner would), but can also be tapped away
// immediately. Naturally never appears once JANAADHAAR_MODE=real, since the
// backend then never sends a sms_preview at all.
const AUTO_DISMISS_MS = 8000;

export default function SmsPopup({ text, onDismiss }) {
  useEffect(() => {
    if (!text) return undefined;
    const timer = setTimeout(onDismiss, AUTO_DISMISS_MS);
    return () => clearTimeout(timer);
  }, [text, onDismiss]);

  if (!text) return null;

  return (
    <div
      onClick={onDismiss}
      role="button"
      aria-label="Dismiss SMS notification"
      style={{
        position: "absolute",
        top: 10,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 20,
        display: "flex",
        alignItems: "center",
        gap: 10,
        maxWidth: "88%",
        padding: "10px 14px",
        borderRadius: 14,
        background: "#1f2937",
        color: "#fff",
        boxShadow: "0 6px 18px rgba(0,0,0,0.25)",
        cursor: "pointer",
        animation: "sms-popup-slide-in 0.25s ease-out",
      }}
    >
      <span style={{ fontSize: 20, flexShrink: 0 }}>💬</span>
      <span style={{ fontSize: 13, lineHeight: 1.4 }}>{text}</span>
    </div>
  );
}
