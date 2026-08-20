// Minimal hand-rolled markdown rendering - generate_node's replies come out
// as markdown (headers, bold, italic, inline code, links, lists, quotes).
// Everything is built as React elements - no dangerouslySetInnerHTML, no
// rehype-raw - so model output can never inject markup. Links are limited
// to http(s)/mailto/tel and open in a new tab with rel="noopener".

const SAFE_URL = /^(https?:\/\/|mailto:|tel:)/i;

// Inline tokens, in precedence order: code, link, bold, italic.
const INLINE_RE = /(`[^`\n]+`|\[[^\]\n]+\]\((?:https?:\/\/|mailto:|tel:)[^\s)]+\)|\*\*[^*\n]+?\*\*|(?<![\w*])\*(?!\s)[^*\n]+?(?<!\s)\*(?![\w*])|(?<![\w_])_(?!\s)[^_\n]+?(?<!\s)_(?![\w_]))/g;

function renderInline(text, keyPrefix) {
  const out = [];
  let last = 0;
  let i = 0;
  for (const m of text.matchAll(INLINE_RE)) {
    if (m.index > last) out.push(<span key={`${keyPrefix}-t${i++}`}>{text.slice(last, m.index)}</span>);
    const tok = m[0];
    const key = `${keyPrefix}-m${i++}`;
    if (tok.startsWith("`")) {
      out.push(
        <code
          key={key}
          style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: "0.92em", background: "rgba(0,0,0,0.06)", padding: "1px 4px", borderRadius: 4 }}
        >
          {tok.slice(1, -1)}
        </code>,
      );
    } else if (tok.startsWith("[")) {
      const close = tok.indexOf("](");
      const label = tok.slice(1, close);
      const href = tok.slice(close + 2, -1);
      if (SAFE_URL.test(href)) {
        out.push(
          <a key={key} href={href} target="_blank" rel="noopener noreferrer" style={{ color: "inherit", textDecoration: "underline" }}>
            {renderInline(label, key)}
          </a>,
        );
      } else {
        out.push(<span key={key}>{label}</span>);
      }
    } else if (tok.startsWith("**")) {
      out.push(<strong key={key}>{renderInline(tok.slice(2, -2), key)}</strong>);
    } else {
      out.push(<em key={key}>{tok.slice(1, -1)}</em>);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(<span key={`${keyPrefix}-t${i++}`}>{text.slice(last)}</span>);
  return out;
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
    if (/^>\s?/.test(trimmed)) {
      return (
        <div
          key={i}
          style={{ paddingLeft: 10, marginBottom: 2, borderLeft: "3px solid #d1d5db", color: "#4b5563", fontStyle: "italic" }}
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

export default function MessageBubble({ role, text, onRetry }) {
  const isUser = role === "user";
  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", marginBottom: 10 }}>
      <div
        style={{
          maxWidth: "80%",
          padding: "10px 14px",
          borderRadius: 14,
          fontSize: 14,
          lineHeight: 1.45,
          whiteSpace: "pre-wrap",
          overflowWrap: "anywhere",
          background: isUser ? "#1a56db" : "#ffffff",
          color: isUser ? "#ffffff" : "#1f2937",
          border: isUser ? "none" : "1px solid #d1d5db",
        }}
      >
        {renderText(text)}
        {onRetry && (
          <div style={{ marginTop: 8 }}>
            <button
              onClick={onRetry}
              aria-label="Retry"
              style={{ fontSize: 12, padding: "4px 10px", borderRadius: 12, border: "1px solid #1a56db", background: "#fff", color: "#1a56db", cursor: "pointer" }}
            >
              Phir se bhejein / Retry
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
