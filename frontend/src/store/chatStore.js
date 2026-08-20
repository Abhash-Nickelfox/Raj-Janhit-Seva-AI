import { create } from "zustand";
import { sendMessage as apiSendMessage } from "../api/chatApi.js";

const SESSION_KEY = "janseva.sessionId";

function newSessionId() {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `sess-${Math.random().toString(36).slice(2)}`;
}

// Persisted in sessionStorage so a page reload keeps the same backend
// conversation (the server keeps state per session_id until its TTL);
// a new tab still gets a fresh session.
function loadSessionId() {
  try {
    const stored = window.sessionStorage.getItem(SESSION_KEY);
    if (stored && /^[A-Za-z0-9_-]{1,64}$/.test(stored)) return stored;
    const fresh = newSessionId();
    window.sessionStorage.setItem(SESSION_KEY, fresh);
    return fresh;
  } catch {
    return newSessionId();
  }
}

const WELCOME_TEXT =
  "नमस्ते! मैं राजस्थान सरकारी योजनाओं और नौकरियों में मदद करता हूं।\n" +
  "Namaste! Main Rajasthan sarkari yojnaon aur naukriyon mein madad karta hoon.\n" +
  "Hi! I help with Rajasthan govt schemes and jobs.";

const TIMEOUT_TEXT = "Jawab aane mein zyada samay lag gaya. Kripya phir se try karein.";
const GENERIC_ERROR_TEXT = "Maaf kijiye, kuch dikkat aa gayi. Kripya phir se try karein.";
const BUSY_TEXT = "Pichhle sandesh ka jawab aa raha hai - kripya ek pal rukein, phir bolein.";

export const useChatStore = create((set, get) => ({
  sessionId: loadSessionId(),
  messages: [{ role: "bot", text: WELCOME_TEXT }],
  isLoading: false,
  // Always null now that LanguageSelector is removed; the backend still
  // accepts it to lock reply_lang for a session.
  preferredLang: null,
  // Transient note shown near the input (e.g. a voice message rejected
  // because a reply is still loading). Cleared on the next send.
  notice: null,
  clearNotice: () => set({ notice: null }),

  // Mock-only: OTP preview text the turn the Jan Aadhaar mock generates one
  // (only when the server runs with JANAADHAAR_DEMO_OTP_PREVIEW=1).
  smsPopup: null,
  dismissSmsPopup: () => set({ smsPopup: null }),

  // Retry the last failed user message (the error bubble offers this).
  retryLast: () => {
    const last = get().lastFailedText;
    if (!last || get().isLoading) return;
    set((state) => ({
      messages: state.messages.filter((m) => !m.retryable),
      lastFailedText: null,
    }));
    get().sendMessage(last, { source: "retry" });
  },
  lastFailedText: null,

  sendMessage: async (text, { source = "text" } = {}) => {
    const trimmed = (text || "").trim();
    if (!trimmed) return;
    if (get().isLoading) {
      // Voice sends arrive asynchronously (after transcription) and can
      // land while a reply is still loading - reject visibly rather than
      // silently dropping what the citizen just said.
      if (source === "voice") set({ notice: BUSY_TEXT });
      return;
    }

    set((state) => ({
      // On retry the citizen's bubble is already on screen - don't add it twice.
      messages: source === "retry" ? state.messages : [...state.messages, { role: "user", text: trimmed }],
      isLoading: true,
      notice: null,
      lastFailedText: null,
    }));

    try {
      const { reply, sms_preview, retryable } = await apiSendMessage(
        get().sessionId,
        trimmed,
        get().preferredLang,
      );
      set((state) => ({
        messages: [...state.messages, { role: "bot", text: reply, retryable: Boolean(retryable) }],
        isLoading: false,
        smsPopup: sms_preview || null,
        lastFailedText: retryable ? trimmed : null,
      }));
    } catch (err) {
      const timedOut = err?.name === "TimeoutError" || err?.name === "AbortError";
      set((state) => ({
        messages: [
          ...state.messages,
          { role: "bot", text: timedOut ? TIMEOUT_TEXT : GENERIC_ERROR_TEXT, retryable: true },
        ],
        isLoading: false,
        lastFailedText: trimmed,
      }));
    }
  },
}));
