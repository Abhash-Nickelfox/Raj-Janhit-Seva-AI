export default async function run(page, ui) {
  const initial = await ui.snapshot();
  const initialText = await page.textContent("body");

  const inputRef = initial.match(/@(e\d+) textbox/)?.[1];
  const sendRef = initial.match(/@(e\d+) button "Send"/)?.[1];

  if (!inputRef || !sendRef) {
    return { error: "could not find input/send controls", snapshot: initial };
  }

  await ui.fill(`@${inputRef}`, "widow pension");
  await ui.click(`@${sendRef}`);

  // wait for a new bot bubble to appear (bubble count increases beyond the
  // initial welcome message) rather than a fixed delay
  await page.waitForFunction(
    (before) => document.body.innerText.length > before,
    initialText.length + 20,
    { timeout: 20000 }
  );

  await page.waitForTimeout(500);
  const finalText = await page.textContent("body");

  return {
    hasHeader: initialText.includes("जन सेवा AI"),
    hasWelcome: initialText.includes("Namaste"),
    hasQuickReplies: initialText.includes("पेंशन") || initialText.includes("छात्रवृत्ति"),
    finalBodyTextTail: finalText.slice(-800),
  };
}
