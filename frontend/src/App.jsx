import ChatWindow from "./components/ChatWindow.jsx";
import sideLeft from "./assets/figma/side-illustration-left.png";
import sideRight from "./assets/figma/side-illustration-right.png";

export default function App() {
  return (
    <div
      style={{
        position: "relative",
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        background: "var(--color-bg)",
        padding: "34px 20px 44px",
        overflow: "hidden",
      }}
    >
      <img
        src={sideLeft}
        alt=""
        style={{
          position: "fixed",
          left: -40,
          bottom: 0,
          width: "auto",
          height: "auto",
          maxWidth: "clamp(340px, calc((100vw - 816px) / 2 + 60px), 1020px)",
          maxHeight: "97vh",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />
      <img
        src={sideRight}
        alt=""
        style={{
          position: "fixed",
          right: 40,
          bottom: -30,
          width: "auto",
          height: "auto",
          maxWidth: "clamp(380px, calc((100vw - 816px) / 2 + 100px), 1120px)",
          maxHeight: "97vh",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />
      <div style={{ position: "relative", width: "100%", maxWidth: 776, zIndex: 1 }}>
        <ChatWindow />
      </div>
    </div>
  );
}
