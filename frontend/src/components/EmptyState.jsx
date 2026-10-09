export default function EmptyState() {
  return (
    <div style={{ padding: "28px 24px 20px" }}>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 4,
          paddingTop: 12,
          maxWidth: 570,
        }}
      >
        <p
          style={{
            margin: 0,
            fontFamily: "var(--font-devanagari)",
            fontWeight: 700,
            fontSize: 17.5,
            lineHeight: "27.936px",
            color: "var(--color-ink)",
          }}
        >
          नमस्ते! मैं राजस्थान सरकारी योजनाओं और नौकरियों में मदद करता हूं।
        </p>
        <p
          style={{
            margin: 0,
            fontFamily: "var(--font-latin)",
            fontWeight: 400,
            fontSize: 15.52,
            lineHeight: "25.142px",
            letterSpacing: -0.051,
            color: "var(--color-ink-muted)",
          }}
        >
          Namaste! Main Rajasthan sarkari yojnaon aur naukriyon mein madad karta hoon.
        </p>
        <p
          style={{
            margin: 0,
            fontFamily: "var(--font-latin)",
            fontWeight: 400,
            fontSize: 15.52,
            lineHeight: "25.142px",
            letterSpacing: -0.051,
            color: "var(--color-ink-muted)",
          }}
        >
          Hi! I help with Rajasthan govt schemes and jobs.
        </p>
      </div>
    </div>
  );
}
