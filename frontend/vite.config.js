import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-time proxy so the frontend can call relative paths (/chat, /health)
// without CORS issues, matching how it would sit behind a single origin
// in production too.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      "/chat": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
