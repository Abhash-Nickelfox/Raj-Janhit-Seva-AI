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
    // `true` is deliberately permissive for temporary testing over a tunnel
    // (ngrok etc.) in dev ONLY - this block is the dev server config and is
    // not used by `vite build`. Tighten to an explicit list (e.g.
    // ["localhost", ".ngrok-free.app"]) before exposing the dev server to
    // anything public.
    allowedHosts: true,
  },
});
