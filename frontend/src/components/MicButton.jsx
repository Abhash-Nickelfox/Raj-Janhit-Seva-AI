import { useRef, useState } from "react";
import { transcribeAudio } from "../api/chatApi.js";

// Press to start recording (records via MediaRecorder), stops ONLY when the
// citizen taps the mic icon again - no auto-stop of any kind (2026-08-20,
// removing an earlier silence-detection + 30-second-cap auto-stop, per
// explicit direction: the mic should never close on its own, only on a
// manual tap, the same way it works elsewhere). Transcribes via Sarvam's
// Saaras (app/stt_service.py), then auto-sends the transcribed text exactly
// like a typed message (no review-before-send step, by design). Errors
// surface via onError rather than failing silently, so the citizen always
// knows to fall back to typing.
//
// Known, separate limitation this doesn't solve: Sarvam's synchronous REST
// STT endpoint is documented for files under 30 seconds - a citizen who
// talks past that may get a failed/rejected transcription regardless of
// this component's own stop behavior. That's a server-side API constraint,
// not something a frontend auto-stop timer was ever really fixing (it was
// just cutting recordings short before hitting it) - if long recordings
// need to actually work, that's a separate fix (Sarvam's batch/streaming
// API instead of the sync REST one), not something to solve here.
export default function MicButton({ onSend, disabled, onError }) {
  const [status, setStatus] = useState("idle"); // idle | recording | transcribing
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const supported = typeof navigator !== "undefined" && navigator.mediaDevices && window.MediaRecorder;
  if (!supported) return null; // graceful degrade - text input still works fine without this

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setStatus("transcribing");
        try {
          const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
          const text = (await transcribeAudio(blob)).trim();
          if (text) {
            onSend(text);
          } else {
            onError?.("Kuch sunai nahi diya. Kripya phir se try karein ya type karein.");
          }
        } catch (err) {
          onError?.("Awaaz samajhne mein dikkat aayi. Kripya type karke bhejein.");
        } finally {
          setStatus("idle");
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setStatus("recording");
    } catch (err) {
      onError?.("Microphone ka access nahi mila. Kripya permission dein ya type karke bhejein.");
      setStatus("idle");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
  };

  const handleClick = () => {
    if (status === "idle") startRecording();
    else if (status === "recording") stopRecording(); // the only way recording ever stops
    // a click during "transcribing" is ignored - nothing to start/stop
  };

  const isRecording = status === "recording";
  const isBusy = status === "transcribing";

  return (
    <button
      onClick={handleClick}
      disabled={disabled || isBusy}
      aria-label={isRecording ? "Stop recording" : "Record a voice message"}
      title={isRecording ? "Recording... click to stop" : "Click to speak"}
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
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M12 14a3 3 0 003-3V6a3 3 0 10-6 0v5a3 3 0 003 3zm5-3a5 5 0 01-10 0H5a7 7 0 006 6.92V21h2v-3.08A7 7 0 0019 11h-2z"
          fill="currentColor"
        />
      </svg>
    </button>
  );
}
