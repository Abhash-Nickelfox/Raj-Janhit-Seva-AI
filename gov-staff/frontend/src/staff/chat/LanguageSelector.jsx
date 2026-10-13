// Copied from the citizen app (frontend/src/components) so staff chat looks the same.
import { useState } from "react";
import iconChevron from "../../assets/figma/icon-chevron.svg";

// Re-adds a language picker in front of the store's existing preferredLang
// plumbing (chatStore.js / chatApi.js already send it on every /chat call -
// it just had no UI since the old LanguageSelector was removed). Picking one
// here locks reply_lang for the rest of the session, same as before.
const OPTIONS = [
  { value: null, label: "Auto" },
  { value: "english", label: "English" },
  { value: "hindi", label: "हिंदी" },
  { value: "hinglish", label: "Hinglish" },
];

export default function LanguageSelector({ value, onChange, disabled }) {
  const [open, setOpen] = useState(false);
  const current = OPTIONS.find((o) => o.value === value) || OPTIONS[0];

  return (
    <div style={{ position: "relative" }}>
      <button
        onClick={() => setOpen((o) => !o)}
        disabled={disabled}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 7,
          minHeight: 40,
          padding: "0 15px",
          borderRadius: 999,
          border: "1px solid var(--color-card-border)",
          background: "transparent",
          cursor: disabled ? "default" : "pointer",
          flexShrink: 0,
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-latin)",
            fontWeight: 500,
            fontSize: 14.4,
            color: "var(--color-ink-muted)",
            whiteSpace: "nowrap",
          }}
        >
          {current.label}
        </span>
        <img
          src={iconChevron}
          alt=""
          width={13}
          height={13}
          style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform 0.15s" }}
        />
      </button>

      {open && (
        <>
          <div
            onClick={() => setOpen(false)}
            style={{ position: "fixed", inset: 0, zIndex: 30 }}
          />
          <div
            style={{
              position: "absolute",
              bottom: "calc(100% + 6px)",
              left: 0,
              background: "#fff",
              border: "1px solid var(--color-card-border)",
              borderRadius: 12,
              boxShadow: "0px 4px 16px rgba(17,26,46,0.12)",
              overflow: "hidden",
              zIndex: 31,
              minWidth: 140,
            }}
          >
            {OPTIONS.map((opt) => (
              <button
                key={String(opt.value)}
                onClick={() => {
                  onChange(opt.value);
                  setOpen(false);
                }}
                style={{
                  display: "block",
                  width: "100%",
                  textAlign: "left",
                  padding: "10px 14px",
                  border: "none",
                  background: opt.value === value ? "var(--color-accent-blue-bg)" : "transparent",
                  color: opt.value === value ? "var(--color-accent-blue)" : "var(--color-ink)",
                  fontFamily: "var(--font-latin)",
                  fontSize: 14,
                  cursor: "pointer",
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
