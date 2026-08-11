import ChatWindow from "./components/ChatWindow.jsx";

export default function App() {
  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "stretch",
        background: "#e5e7eb",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 480,
          display: "flex",
          flexDirection: "column",
          background: "#ffffff",
          boxShadow: "0 0 20px rgba(0,0,0,0.1)",
        }}
      >
        <ChatWindow />
      </div>
    </div>
  );
}
