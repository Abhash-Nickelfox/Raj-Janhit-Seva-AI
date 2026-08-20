(async () => {
  const initialLen = document.body.innerText.length;
  const input = document.querySelector('input[type="text"]');
  const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  nativeSetter.call(input, 'widow pension');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  const buttons = Array.from(document.querySelectorAll('button'));
  const sendBtn = buttons.find(b => b.getAttribute('aria-label') === 'Send');
  sendBtn.click();
  const start = Date.now();
  while (document.body.innerText.length <= initialLen + 20 && Date.now() - start < 20000) {
    await new Promise(r => setTimeout(r, 300));
  }
  await new Promise(r => setTimeout(r, 500));
  return document.body.innerText.slice(-600);
})()
