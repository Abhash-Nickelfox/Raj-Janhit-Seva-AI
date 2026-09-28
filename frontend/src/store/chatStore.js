import { create } from "zustand";
import { sendMessage as apiSendMessage } from "../api/chatApi.js";

function newSessionId() {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `sess-${Math.random().toString(36).slice(2)}`;
}

const WELCOME_TEXT =
  "नमस्ते! मैं राजस्थान सरकारी योजनाओं और नौकरियों में मदद करता हूं।\n" +
  "Namaste! Main Rajasthan sarkari yojnaon aur naukriyon mein madad karta hoon.\n" +
  "Hi! I help with Rajasthan govt schemes and jobs.";

// Unlike the reference frontend, this doesn't fire an empty-message request
// on mount to fetch a greeting from the backend - the conversation graph's
// extraction nodes expect a real citizen message, not an empty kickoff, so
// the welcome text is just shown directly instead of round-tripping to the
// server for it.
export const useChatStore = create((set, get) => ({
  sessionId: newSessionId(),
  messages: [{ role: "bot", text: WELCOME_TEXT }],
  isLoading: false,
  // Sent on every /chat request (see api/chatApi.js) - the backend LOCKS
  // reply_lang to this for the rest of the session once set. null means
  // per-turn auto-detection decides the reply's language instead.
  preferredLang: null,
  setPreferredLang: (lang) => set({ preferredLang: lang }),

  // Mock-only (2026-08-17): set to the OTP preview text the turn the Jan
  // Aadhaar mock generates one, so SmsPopup can show it as a simulated
  // incoming SMS instead of requiring a look at server logs. null otherwise,
  // and cleared once the popup dismisses itself.
  smsPopup: null,
  dismissSmsPopup: () => set({ smsPopup: null }),

  sendMessage: async (text) => {
    const trimmed = text.trim();
    if (!trimmed || get().isLoading) return;

    set((state) => ({
      messages: [...state.messages, { role: "user", text: trimmed }],
      isLoading: true,
    }));

    try {
      const { reply, sms_preview } = await apiSendMessage(get().sessionId, trimmed, get().preferredLang);
      set((state) => ({
        messages: [...state.messages, { role: "bot", text: reply }],
        isLoading: false,
        smsPopup: sms_preview || null,
      }));
    } catch (err) {
      set((state) => ({
        messages: [
          ...state.messages,
          { role: "bot", text: "Maaf kijiye, kuch dikkat aa gayi. Kripya phir se try karein." },
        ],
        isLoading: false,
      }));
    }
  },
}));
