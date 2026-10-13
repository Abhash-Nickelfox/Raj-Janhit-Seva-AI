import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));
const CITIZEN_API = process.env.CITIZEN_API_TARGET || "http://localhost:8000";
const STAFF_API = process.env.STAFF_API_TARGET || "http://localhost:8200";
const citizenSrc = path.resolve(here, "../../frontend/src");

// One frontend for both assistants (port 5180):
//   /citizen  -> the citizen chat app, used read-only from ../../frontend/src
//                (never edited here); its /chat, /transcribe, /health calls go
//                to the citizen backend (8000).
//   /staff/*  -> staff register/login/chat; its /api calls go to the staff
//                backend (8200).
// Same-origin proxying means no CORS and the staff session cookie stays first-party.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@citizen": citizenSrc },
    // the citizen files must use this app's single copy of React/zustand
    dedupe: ["react", "react-dom", "zustand"],
  },
  server: {
    port: 5180,
    strictPort: true,
    fs: { allow: [here, citizenSrc] },
    proxy: {
      "/chat": CITIZEN_API,
      "/transcribe": CITIZEN_API,
      "/health": CITIZEN_API,
      "/api": STAFF_API,
    },
  },
  preview: { port: 5180, strictPort: true },
});
