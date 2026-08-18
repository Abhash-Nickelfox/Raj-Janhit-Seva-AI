import { useRef, useState } from "react";
import { transcribeAudio } from "../api/chatApi.js";

// 0-128 scale (time-domain byte data is centered at 128) - the RMS level
// below which the mic is considered "quiet". Tuned for a normal room, not
// a noisy one - if this proves too sensitive/insensitive in practice, this
// is the number to adjust.
const SILENCE_RMS_THRESHOLD = 8;
// How long the citizen must stay quiet, AFTER having spoken at least once,
// before recording auto-stops - long enough to survive a natural pause
// mid-sentence, short enough to feel responsive once they're actually done.
const SILENCE_DURATION_MS = 1500;
// Safety cap regardless of silence detection, in case background noise or
// a quiet mic never registers as "silence" - recording still stops on its
// own rather than running indefinitely.
const MAX_RECORDING_MS = 30000;

// Press to start recording (records via MediaRecorder), auto-stops once
// voice-activity detection notices the citizen has gone quiet after
// speaking, transcribes via Sarvam's Saaras (app/stt_service.py), then
// auto-sends the transcribed text exactly like a typed message (no
// review-before-send step, by design). A manual click while recording
// still stops it immediately too, as a fallback/override. Errors surface
// via onError rather than failing silently, so the citizen always knows
// to fall back to typing.
export default function MicButton({ onSend, disabled, onError, lastReplyLang }) {
  const [status, setStatus] = useState("idle"); // idle | recording | transcribing
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const audioCtxRef = useRef(null);
  const vadFrameRef = useRef(null);
  const maxTimerRef = useRef(null);

  const supported = typeof navigator !== "undefined" && navigator.mediaDevices && window.MediaRecorder;
  if (!supported) return null; // graceful degrade - text input still works fine without this

  const stopSilenceDetection = () => {
    if (vadFrameRef.current !== null) cancelAnimationFrame(vadFrameRef.current);
    vadFrameRef.current = null;
    if (maxTimerRef.current !== null) clearTimeout(maxTimerRef.current);
    maxTimerRef.current = null;
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
  };

  // Watches the mic's volume so recording stops itself once the citizen
  // finishes speaking, instead of requiring a second click. Only starts
  // counting silence AFTER real speech has been heard, so a citizen who
  // pauses to think before speaking at all isn't cut off immediately.
  const startSilenceDetection = (stream, onSilenceDetected) => {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContextClass();
    audioCtxRef.current = audioCtx;
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);

    const data = new Uint8Array(analyser.frequencyBinCount);
    let hasSpoken = false;
    let silenceStartedAt = null;

    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sumSquares = 0;
      for (let i = 0; i < data.length; i++) {
        const deviation = data[i] - 128;
        sumSquares += deviation * deviation;
      }
      const rms = Math.sqrt(sumSquares / data.length);

      if (rms > SILENCE_RMS_THRESHOLD) {
        hasSpoken = true;
        silenceStartedAt = null;
      } else if (hasSpoken) {
        if (silenceStartedAt === null) {
          silenceStartedAt = performance.now();
        } else if (performance.now() - silenceStartedAt > SILENCE_DURATION_MS) {
          onSilenceDetected();
          return; // stop polling - the recording is being stopped
        }
      }
      vadFrameRef.current = requestAnimationFrame(tick);
    };
    vadFrameRef.current = requestAnimationFrame(tick);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stopSilenceDetection();
        stream.getTracks().forEach((track) => track.stop());
        setStatus("transcribing");
        try {
          const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
          const text = (await transcribeAudio(blob, lastReplyLang)).trim();
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

      const autoStop = () => {
        if (mediaRecorderRef.current?.state === "recording") {
          mediaRecorderRef.current.stop();
        }
      };
      startSilenceDetection(stream, autoStop);
      maxTimerRef.current = setTimeout(autoStop, MAX_RECORDING_MS);
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
    else if (status === "recording") stopRecording(); // manual override - still works alongside auto-stop
    // a click during "transcribing" is ignored - nothing to start/stop
  };

  const isRecording = status === "recording";
  const isBusy = status === "transcribing";

  return (
    <button
      onClick={handleClick}
      disabled={disabled || isBusy}
      aria-label={isRecording ? "Stop recording" : "Record a voice message"}
      title={isRecording ? "Recording... stops automatically when you finish speaking, or click to stop now" : "Click to speak"}
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
