// Minimal English/Hindi switch. Every label lives in both en.json and hi.json.
import { createContext, useContext, useMemo, useState } from "react";

import en from "./en.json";
import hi from "./hi.json";

const DICTS = { en, hi };
const LanguageContext = createContext(null);

function lookup(dict, key) {
  return key.split(".").reduce((node, part) => (node && typeof node === "object" ? node[part] : undefined), dict);
}

export function LanguageProvider({ children }) {
  const [lang, setLang] = useState(() => {
    let stored = "en";
    try {
      stored = localStorage.getItem("staff.lang") === "hi" ? "hi" : "en";
    } catch {
      /* storage blocked - default to English */
    }
    document.documentElement.lang = stored;
    return stored;
  });
  const value = useMemo(() => {
    const t = (key, vars = {}) => {
      const text = lookup(DICTS[lang], key) ?? lookup(en, key) ?? key;
      return String(text).replace(/\{\{(\w+)\}\}/g, (_, v) => vars[v] ?? "");
    };
    const change = (next) => {
      setLang(next);
      try {
        localStorage.setItem("staff.lang", next);
      } catch {
        /* private mode - keep in memory only */
      }
      document.documentElement.lang = next;
    };
    return { lang, setLang: change, t };
  }, [lang]);
  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export const useT = () => useContext(LanguageContext);
