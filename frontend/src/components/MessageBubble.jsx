// Minimal hand-rolled markdown rendering - generate_node's replies come out
// as markdown (headers, bold, lists), and without this they'd show up as
// literal "#"/"**"/"-" characters in the chat bubble.
function renderInline(text, keyPrefix) {
  const parts = text.split(/(\*\*.+?\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>;
    }
    return <span key={`${keyPrefix}-${i}`}>{part}</span>;
  });
}

function renderText(text) {
  const lines = text.split("\n");
  return lines.map((line, i) => {
    const trimmed = line.trim();
    if (/^#{1,3}\s/.test(trimmed)) {
      const content = trimmed.replace(/^#{1,3}\s/, "");
      return (
        <div key={i} style={{ fontWeight: 700, marginTop: i === 0 ? 0 : 8, marginBottom: 4 }}>
          {renderInline(content, i)}
        </div>
      );
    }
    if (/^[-*]\s/.test(trimmed)) {
      return (
        <div key={i} style={{ paddingLeft: 12, marginBottom: 2 }}>
          {"• "}
          {renderInline(trimmed.replace(/^[-*]\s/, ""), i)}
        </div>
      );
    }
    if (/^\d+\.\s/.test(trimmed)) {
      return (
        <div key={i} style={{ paddingLeft: 12, marginBottom: 2 }}>
          {renderInline(trimmed, i)}
        </div>
      );
    }
    // Quoted scheme eligibility text (generate_node sometimes cites the
    // source verbatim with a leading ">") - without this case it fell
    // through to the plain-text default and showed a literal ">" character.
    if (/^>\s?/.test(trimmed)) {
      return (
        <div
          key={i}
          style={{
            paddingLeft: 10,
            marginBottom: 2,
            borderLeft: "3px solid #d1d5db",
            color: "#4b5563",
            fontStyle: "italic",
          }}
        >
          {renderInline(trimmed.replace(/^>\s?/, ""), i)}
        </div>
      );
    }
    if (trimmed === "" || trimmed === "---") {
      return <div key={i} style={{ height: 6 }} />;
    }
    return (
      <div key={i} style={{ marginBottom: 2 }}>
        {renderInline(line, i)}
      </div>
    );
  });
}

export default function MessageBubble({ role, text }) {
  const isUser = role === "user";
  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", marginBottom: 10 }}>
      <div
        style={{
          maxWidth: "80%",
          padding: "10px 14px",
          borderRadius: 14,
          fontFamily: "var(--font-latin)",
          fontSize: 14,
          lineHeight: 1.45,
          whiteSpace: "pre-wrap",
          background: isUser ? "var(--color-send-bg)" : "#ffffff",
          color: isUser ? "#ffffff" : "var(--color-ink)",
          border: isUser ? "none" : "1px solid var(--color-card-border)",
        }}
      >
        {renderText(text)}
      </div>
    </div>
  );
}
