# Recording the hackathon demo (about 3 minutes)

## 1. Record a real call on your Mac (best: real Whisper + Qwen + Kokoro)
```bash
ollama serve &                                   # qwen2.5:7b pulled
python -m swiftroute.voice.bot                   # terminal 1
# browser tab A: http://localhost:7860           (Pipecat playground, click Connect)
# browser tab B: http://localhost:7860/inspector (watch it live)
```
Script to say (shows the most mechanisms in one call):
1. "Hi, I missed my delivery. Can you bring it on Saturday?"
2. Read the tracking number **with one letter wrong** (e.g. say D instead of B), spelling one letter as "R for Romeo".
3. Say "yes" **before** the read-back finishes (barge-in).
4. Postal code in two pieces: "V8A" … pause … "1L1" (slot merges it, "Mm-hm, go on.").
5. When told Saturday isn't available: "Okay, Monday then."
6. "Can I talk to a real person?" (allowed transfer), or "Actually my name is Anmol" (no transfer).

The trace and a .wav land in `data/traces/`. Bake it into a single shareable page:
```bash
python -m swiftroute.inspector.build data/traces/<that-call>.jsonl
```

## 2. No mic? Use the simulation
```bash
python -m swiftroute.inspector.simulate --llm ollama --stt whisper   # real models on your Mac
python -m swiftroute.inspector.build data/traces/<newest>.jsonl
```

## 3. Screen-record walkthrough (talk track)
| time | show | say |
|---|---|---|
| 0:00 | Inspector, press Play | "A phone agent for a parcel courier, fully open-source. Every step it takes is on screen." |
| 0:20 | Spectrogram + pipeline strip | "This is what Whisper hears. Orange bars are VAD; green dots are smart-turn deciding the caller is done." |
| 0:45 | Turn 2 → ID decoder | "Whisper got one letter wrong. 874 guesses, the checksum kills all but 10, one DB lookup leaves 1. It reads back only from the changed character." |
| 1:10 | Barge-in mark | "Caller said yes mid read-back. The unplayed audio is dropped and the model is told in CALL STATE, never in its own words." |
| 1:30 | Postal turn: smart-turn says *not done*, checksum ends the turn | "The audio model wanted to wait. The code knew the postal code was complete." |
| 1:50 | Reschedule turn → prompt viewer, tool result DENIED | "Here's the exact prompt. The model asked for Saturday; policy.py said no. The model can explain a verdict, never override it." |
| 2:15 | Latency waterfall + chart | "Voice-to-voice per turn, broken into turn detection, agent, TTS. Zero LLM calls on code turns." |
| 2:35 | "What it handles" tab → Show in call | "Every one of these came from a real failure, with a regression test." |
| 2:50 | "How it gets better" | "No magic self-learning: failure, structural fix, test, eval, and next, distilling 7B into 1.5B." |
