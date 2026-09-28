import logoFull from "../assets/figma/logo-full.png";

export default function Header() {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 2,
        minWidth: 0,
        padding: "16px 20px",
        borderBottom: "1px solid var(--color-card-border)",
      }}
    >
      <div
        style={{
          background: "#fff",
          padding: "4px 8px",
          borderRadius: 7.072,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        <img src={logoFull} alt="Raj Janhit Sewa" height={64} style={{ display: "block", width: "auto" }} />
      </div>

      <div style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
        <p
          style={{
            margin: 0,
            fontFamily: "var(--font-devanagari)",
            fontWeight: 700,
            fontSize: 16,
            lineHeight: "20.8px",
            color: "var(--color-ink)",
            whiteSpace: "nowrap",
          }}
        >
          राज जनहित सेवा<span style={{ color: "var(--color-accent-orange)" }}>.ai</span>
        </p>
        <p
          style={{
            margin: 0,
            fontFamily: "var(--font-latin)",
            fontWeight: 400,
            fontSize: 12.8,
            lineHeight: "20.736px",
            letterSpacing: -0.051,
            color: "var(--color-ink-faint)",
          }}
        >
          Rajasthan Government Scheme Assistant
        </p>
      </div>

      <div style={{ flex: 1, minWidth: 0, display: "flex", justifyContent: "flex-end" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: 3.5,
              background: "var(--color-accent-green)",
              flexShrink: 0,
            }}
          />
          <span
            style={{
              fontFamily: "var(--font-latin)",
              fontWeight: 600,
              fontSize: 12.8,
              letterSpacing: -0.051,
              color: "var(--color-accent-green)",
              whiteSpace: "nowrap",
            }}
          >
            Online
          </span>
        </div>
      </div>
    </div>
  );
}
