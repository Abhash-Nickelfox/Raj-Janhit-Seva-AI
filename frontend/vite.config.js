import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-time proxy so the frontend can call relative paths (/chat, /health,
// /transcribe) without CORS issues, matching how it would sit behind a
// single origin in production too.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      "/chat": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/transcribe": "http://localhost:8000",
    },
    // Vite's dev server otherwise rejects requests whose Host header it
    // doesn't recognize ("Blocked request. This host is not allowed") - a
    // tunnel domain (ngrok, etc.) always fails that check by default.
    // `true` is fine for temporary testing over a tunnel; don't leave this
    // on for anything actually public/production.
    allowedHosts: true,
  },
});
