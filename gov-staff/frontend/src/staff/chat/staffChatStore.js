// Staff chat state. Same shape as the citizen chat store, but talks to the
// staff backend (POST /api/ask, session cookie) - never the citizen backend.
import { create } from "zustand";

import { api } from "../../api/client";

const newSessionId = () =>
  typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `staff-${Math.random().toString(36).slice(2)}`;

const initial = () => ({ sessionId: newSessionId(), messages: [], isLoading: false, preferredLang: null });

export const useStaffChatStore = create((set, get) => ({
  ...initial(),
  setPreferredLang: (lang) => set({ preferredLang: lang }),
  reset: () => set(initial()),

  sendMessage: async (text) => {
    const trimmed = text.trim();
    if (!trimmed || get().isLoading) return;
    set((s) => ({ messages: [...s.messages, { role: "user", text: trimmed }], isLoading: true }));
    try {
      const { reply } = await api.post("/ask", { message: trimmed, session_id: get().sessionId, preferred_lang: get().preferredLang });
      set((s) => ({ messages: [...s.messages, { role: "bot", text: reply }], isLoading: false }));
    } catch (err) {
      const text = err.status === 401
        ? "Your session has ended. Please sign in again. / आपका सत्र समाप्त हो गया है, कृपया फिर से साइन इन करें।"
        : "Sorry, something went wrong. Please try again. / क्षमा करें, कुछ गड़बड़ हुई। कृपया फिर कोशिश करें।";
      set((s) => ({ messages: [...s.messages, { role: "bot", text }], isLoading: false }));
    }
  },
}));
