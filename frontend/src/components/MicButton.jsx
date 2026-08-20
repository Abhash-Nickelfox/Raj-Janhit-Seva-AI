import { useEffect, useRef, useState } from "react";
import { transcribeAudio } from "../api/chatApi.js";

// Safety cap only - there is NO silence-detection auto-stop any more
// (2026-08-20: it cut citizens off mid-thought and the 30 s cap was too
// short for a full description of their situation). Recording is
// click-to-start / click-to-stop, with a live timer so the citizen can see
// it's still listening.
const MAX_RECORDING_MS = 5 * 60 * 1000;

function formatElapsed(ms) {
  const total = Math.floor(ms / 1000);
  const mm = String(Math.floor(total / 60)).padStart(2, "0");
  const ss = String(total % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

// Press to start recording (MediaRecorder), press again to stop; the clip
// is transcribed via Sarvam (app/stt_service.py) and auto-sent exactly like
// a typed message. Errors surface via onError so the citizen knows to fall
// back to typing.
export default function MicButton({ onSend, disabled, onError }) {
  const [status, setStatus] = useState("idle"); // idle | recording | transcribing
  const [elapsedMs, setElapsedMs] = useState(0);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const maxTimerRef = useRef(null);
  const tickTimerRef = useRef(null);
  const startedAtRef = useRef(0);
  const mountedRef = useRef(true);

  const supported = typeof navigator !== "undefined" && navigator.mediaDevices && window.MediaRecorder;

  const clearTimers = () => {
    if (maxTimerRef.current !== null) clearTimeout(maxTimerRef.current);
    maxTimerRef.current = null;
    if (tickTimerRef.current !== null) clearInterval(tickTimerRef.current);
    tickTimerRef.current = null;
  };

  const releaseStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  };

  // Cleanup on unmount: stop any in-flight recording and release the mic.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      clearTimers();
      const rec = mediaRecorderRef.current;
      if (rec && rec.state !== "inactive") {
        rec.onstop = null; // don't try to transcribe/setState after unmount
        try {
          rec.stop();
        } catch {
          /* already stopped */
        }
      }
      mediaRecorderRef.current = null;
      releaseStream();
    };
  }, []);

  if (!supported) return null; // graceful degrade - text input still works

  const stopRecording = () => {
    const rec = mediaRecorderRef.current;
    if (rec && rec.state === "recording") {
      try {
        rec.stop();
      } catch {
        /* race with an already-finished recorder */
      }
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!mountedRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        clearTimers();
        releaseStream();
        if (!mountedRef.current) return;
        setStatus("transcribing");
        try {
          const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
          const text = (await transcribeAudio(blob)).trim();
          if (!mountedRef.current) return;
          if (text) {
            onSend(text);
          } else {
            onError?.("Kuch sunai nahi diya. Kripya phir se try karein ya type karein.");
          }
        } catch (err) {
          if (!mountedRef.current) return;
          const msg =
            err?.name === "TimeoutError" || err?.name === "AbortError"
              ? "Awaaz bhejne mein zyada samay lag gaya. Kripya phir se try karein ya type karein."
              : "Awaaz samajhne mein dikkat aayi. Kripya type karke bhejein.";
          onError?.(msg);
        } finally {
          if (mountedRef.current) {
            setStatus("idle");
            setElapsedMs(0);
          }
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      startedAtRef.current = Date.now();
      setElapsedMs(0);
      setStatus("recording");

      tickTimerRef.current = setInterval(() => {
        setElapsedMs(Date.now() - startedAtRef.current);
      }, 500);
      maxTimerRef.current = setTimeout(stopRecording, MAX_RECORDING_MS);
    } catch (err) {
      releaseStream();
      onError?.("Microphone ka access nahi mila. Kripya permission dein ya type karke bhejein.");
      if (mountedRef.current) setStatus("idle");
    }
  };

  const handleClick = () => {
    if (status === "idle") startRecording();
    else if (status === "recording") stopRecording();
    // a click during "transcribing" is ignored - nothing to start/stop
  };

  const isRecording = status === "recording";
  const isBusy = status === "transcribing";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
      {isRecording && (
        <span
          aria-live="polite"
          style={{ fontSize: 12, fontVariantNumeric: "tabular-nums", color: "#dc2626", minWidth: 38 }}
        >
          {formatElapsed(elapsedMs)}
        </span>
      )}
      <button
        onClick={handleClick}
        disabled={(disabled && !isRecording) || isBusy}  // Stop must always work while recording
        aria-label={isRecording ? "Stop recording" : "Record a voice message"}
        title={isRecording ? "Recording... click to stop and send" : "Click to speak"}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: 40,
          height: 40,
          padding: 0,
          borderRadius: "50%",
          border: "none",
          background: isRecording ? "#dc2626" : disabled || isBusy ? "#9ca3af" : "#6b7280",
          color: "#fff",
          cursor: disabled || isBusy ? "default" : "pointer",
          flexShrink: 0,
          animation: isRecording ? "mic-pulse 1.2s infinite" : "none",
        }}
      >
        {isRecording ? (
          <svg width="16" height="16" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor" />
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M12 14a3 3 0 003-3V6a3 3 0 10-6 0v5a3 3 0 003 3zm5-3a5 5 0 01-10 0H5a7 7 0 006 6.92V21h2v-3.08A7 7 0 0019 11h-2z"
              fill="currentColor"
            />
          </svg>
        )}
      </button>
    </div>
  );
}
