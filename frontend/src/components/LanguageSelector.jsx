import { useEffect, useRef, useState } from "react";

const OPTIONS = [
  { value: "hindi", label: "हिंदी" },
  { value: "hinglish", label: "Hinglish" },
  { value: "english", label: "English" },
];

// Persistent, always-visible selector (top-right of the header) - unlike an
// initial one-time picker, the citizen can change this at any point in the
// conversation, and whatever they pick stays locked (see ConvState.
// locked_reply_lang) until they change it again here.
export default function LanguageSelector({ current, onSelect }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const currentLabel = OPTIONS.find((o) => o.value === current)?.label || "भाषा";

  return (
    <div ref={rootRef} style={{ position: "relative", flexShrink: 0 }}>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label="Change language"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 4,
          padding: "6px 10px",
          borderRadius: 14,
          border: "1px solid rgba(255,255,255,0.7)",
          background: "rgba(255,255,255,0.15)",
          color: "#fff",
          fontSize: 12,
          fontWeight: 600,
          cursor: "pointer",
          whiteSpace: "nowrap",
        }}
      >
        🌐 {currentLabel} ▾
      </button>
      {open && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 6px)",
            right: 0,
            background: "#ffffff",
            border: "1px solid #d1d5db",
            borderRadius: 10,
            overflow: "hidden",
            zIndex: 10,
            minWidth: 110,
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
          }}
        >
          {OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => {
                onSelect(opt.value);
                setOpen(false);
              }}
              style={{
                display: "block",
                width: "100%",
                textAlign: "left",
                padding: "9px 14px",
                border: "none",
                background: current === opt.value ? "#fff3e6" : "#fff",
                color: "#1f2937",
                fontSize: 13,
                fontWeight: current === opt.value ? 700 : 400,
                cursor: "pointer",
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
