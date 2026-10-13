"""Audio taps for the inspector: a live log-mel spectrogram of both sides of the call,
plus the VAD and TTS events that happen alongside the audio.

The features are the ones Whisper itself consumes: 80 mel bins, 25 ms window, 10 ms
hop, log10 power, shown over an 8-decade dynamic range below the running peak (Whisper
clamps the same way). So the caller lane is literally "what the speech recogniser is
looking at", and you can see why B/D/E/P/T/V get confused: their mel patterns differ
only in a short burst at the start of the letter.

Placement in the pipeline (see bot.py / inspector/simulate.py):
    transport.input -> VAD -> [AudioTap "caller"] -> STT -> ... -> TTS -> [AudioTap "agent"] -> output

TTS produces audio faster than real time, so the agent lane is placed on a playout
clock (when the caller actually hears it), not on generation time. A barge-in
(InterruptionFrame) resets that clock: the rest of the reply is never played.
"""
from __future__ import annotations

import numpy as np
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .. import trace

N_MELS = 80
HOP_MS = 10
FMAX = 8000.0            # phone-band-ish and the same top frequency for both lanes


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS, fmax: float = FMAX) -> np.ndarray:
    fmax = min(fmax, sr / 2)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (bins - lo) / max(mid - lo, 1e-9)
        down = (hi - bins) / max(hi - mid, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
        fb[i] *= 2.0 / max(hi - lo, 1e-9)          # slaney-style area normalisation
    return fb


class LogMel:
    """Streaming log-mel: push PCM16 bytes, get (columns, rms_dbfs) back."""

    def __init__(self, sr: int):
        self.sr = sr
        self.win = int(sr * 0.025)
        self.hop = int(sr * HOP_MS / 1000)
        self.n_fft = int(2 ** np.ceil(np.log2(self.win)))
        self.window = np.hanning(self.win + 1)[:-1].astype(np.float32)
        self.fb = mel_filterbank(sr, self.n_fft)
        self.buf = np.zeros(0, dtype=np.float32)
        self.peak = -6.0                           # running max of log10 mel power

    def push(self, pcm: bytes) -> tuple[np.ndarray, float]:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        self.buf = np.concatenate([self.buf, x])
        n = 0 if len(self.buf) < self.win else 1 + (len(self.buf) - self.win) // self.hop
        if n == 0:
            return np.zeros((0, N_MELS), dtype=np.uint8), _dbfs(rms)
        idx = np.arange(self.win)[None, :] + self.hop * np.arange(n)[:, None]
        frames = self.buf[idx] * self.window
        spec = np.abs(np.fft.rfft(frames, n=self.n_fft)) ** 2
        logmel = np.log10(np.maximum(spec @ self.fb.T, 1e-10))
        self.buf = self.buf[n * self.hop:]
        self.peak = max(self.peak - 0.002 * n, float(logmel.max()))   # slow decay, Whisper-style ceiling
        img = np.clip((logmel - (self.peak - 8.0)) / 8.0, 0.0, 1.0)
        return (img * 255).astype(np.uint8), _dbfs(rms)


def _dbfs(rms: float) -> float:
    return round(20 * np.log10(max(rms, 1e-6)), 1)


class CallAudio:
    """Both sides of the call on one timeline, saved next to the trace as a mono 16 kHz
    WAV so the inspector can play the call in sync with the spectrogram.
    Agent audio is placed where it was heard (playout clock), so a barge-in shows up as
    overlapping speech, the way it sounded."""
    SR = 16000

    def __init__(self):
        self.parts: list[tuple[str, float, np.ndarray]] = []

    def add(self, lane: str, start_ms: float, pcm: bytes, sr: int) -> None:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if sr != self.SR and len(x):
            x = np.interp(np.arange(int(len(x) * self.SR / sr)) * (sr / self.SR), np.arange(len(x)), x)
        self.parts.append((lane, start_ms, x))

    def cut(self, lane: str, at_ms: float) -> None:
        """Barge-in: this lane's audio scheduled after `at_ms` was never played."""
        kept = []
        for ln, start, x in self.parts:
            if ln == lane:
                if start >= at_ms:
                    continue
                end = start + len(x) / self.SR * 1000
                if end > at_ms:
                    x = x[: int((at_ms - start) / 1000 * self.SR)]
            kept.append((ln, start, x))
        self.parts = kept

    def save(self, path) -> None:
        import wave
        if not self.parts:
            return
        n = int(max(s / 1000 * self.SR + len(x) for _, s, x in self.parts)) + 1
        mix = np.zeros(n, dtype=np.float32)
        for _, start, x in self.parts:
            i = max(0, int(start / 1000 * self.SR))
            mix[i:i + len(x)] += x[: n - i]
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.SR)
            w.writeframes(np.clip(mix, -32768, 32767).astype(np.int16).tobytes())


class AudioTap(FrameProcessor):
    """Pass-through processor that streams spectrogram columns and audio events to the trace."""

    def __init__(self, lane: str, batch_ms: int = 100, recorder: CallAudio | None = None, **kwargs):
        super().__init__(**kwargs)
        self.lane = lane                       # "caller" | "agent"
        self.recorder = recorder
        self.batch_cols = batch_ms // HOP_MS
        self._mel: LogMel | None = None
        self._cols: list[np.ndarray] = []
        self._rms: list[float] = []
        self._start_ms: float | None = None    # call time of the first pending column
        self._playout_ms = 0.0                 # agent lane: when the next audio will be heard

    def _now_ms(self) -> float:
        return trace.HUB.now_ms()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if trace.active():
            self._observe(frame)
        await self.push_frame(frame, direction)

    def _observe(self, frame: Frame) -> None:
        # mic audio keeps flowing downstream past the STT, so each lane takes only its own kind
        if isinstance(frame, InputAudioRawFrame if self.lane == "caller" else TTSAudioRawFrame):
            self._audio(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame) and self.lane == "caller":
            trace.emit("vad", state="speech", lane=self.lane, why="Silero: speech probability over threshold")
        elif isinstance(frame, VADUserStoppedSpeakingFrame) and self.lane == "caller":
            trace.emit("vad", state="silence", lane=self.lane,
                       why="Silero: 0.2 s of silence. Smart-turn now decides if the caller is done")
        elif isinstance(frame, InterruptionFrame) and self.lane == "agent":
            now = self._now_ms()
            if self._playout_ms > now + 40:        # agent audio was still playing or queued
                self._flush()
                trace.emit("tts.cut", lane="agent", unplayed_ms=round(self._playout_ms - now),
                           why="caller barged in: the rest of the reply is discarded, never heard")
                if self.recorder:
                    self.recorder.cut(self.lane, now)
                self._playout_ms = now
        elif isinstance(frame, TTSStartedFrame):
            trace.emit("tts.start", lane=self.lane)
        elif isinstance(frame, TTSStoppedFrame):
            self._flush()
            trace.emit("tts.stop", lane=self.lane, heard_until_ms=round(self._playout_ms))
        elif isinstance(frame, TTSTextFrame):
            trace.emit("tts.text", text=frame.text, at_ms=round(max(self._playout_ms, self._now_ms())))

    def _audio(self, frame) -> None:
        if self._mel is None or self._mel.sr != frame.sample_rate:
            self._mel = LogMel(frame.sample_rate)
            trace.emit("audio.format", lane=self.lane, sample_rate=frame.sample_rate,
                       n_mels=N_MELS, hop_ms=HOP_MS, win_ms=25, fmax=min(FMAX, frame.sample_rate / 2))
        dur_ms = len(frame.audio) / 2 / frame.sample_rate * 1000
        if self.lane == "agent":
            start = max(self._playout_ms, self._now_ms())
            self._playout_ms = start + dur_ms
        else:
            start = self._now_ms() - dur_ms
        if self._start_ms is None:
            self._start_ms = start
        if self.recorder:
            self.recorder.add(self.lane, start, frame.audio, frame.sample_rate)
        cols, rms = self._mel.push(frame.audio)
        if len(cols):
            self._cols.append(cols)
            self._rms.append(rms)
        if sum(len(c) for c in self._cols) >= self.batch_cols:
            self._flush()

    def _flush(self) -> None:
        if not self._cols:
            self._start_ms = None
            return
        cols = np.concatenate(self._cols)
        trace.emit("audio.mel", lane=self.lane, start_ms=round(self._start_ms, 1), hop_ms=HOP_MS,
                   rms_dbfs=round(max(self._rms), 1),
                   cols=[bytes(c).hex() for c in cols])       # 80 bytes per 10 ms column, hex
        self._cols, self._rms, self._start_ms = [], [], None
