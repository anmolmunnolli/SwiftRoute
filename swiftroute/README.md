# SwiftRoute

A phone support agent for a (fictional) parcel courier, built fully on open-source models.
Callers can track a parcel, reschedule a missed delivery, change the address, hold it at a
depot, or open a damaged/missing claim.

The focus is the engineering that makes a voice agent safe and fast, not the prompt:
deterministic policy, robust capture of spoken IDs, measured latency, and evals.

## System design

[![SwiftRoute system design](docs/architecture.png)](docs/architecture.png)

Callers reach the voice server (Whisper for speech-to-text, Kokoro for speech) by phone or browser. The
front-desk agent and its helper agents use Qwen 2.5 on Ollama, and every action goes through the safety rules.
Outside AI tools connect over MCP, and an improvement team learns from each call. Components, connections, and
how the MCP client and server talk: [docs/architecture.md](docs/architecture.md).

## Phase 1 — core domain (this commit)

| Module | What it does |
|---|---|
| `swiftroute/ids.py` | Tracking-ID format with a Luhn mod-34 check char; spoken-transcript parser (letter names, NATO, "B as in bravo", "double five"); confusion-aware resolver; read-back that covers the corrected characters; position-aware Canadian postal-code repair |
| `swiftroute/policy.py` | Pure-function rules: identity levels, address-change, reschedule, depot hold, claims. Returns a verdict + reason code; the LLM can only explain it |
| `swiftroute/tools.py` | The tool surface the LLM will call, with per-call session state (confirmed parcel, verification level) held outside the model, an audit row per call, and OpenAI-style tool schemas |
| `swiftroute/simulator.py` | Moves parcels through the lifecycle; listeners fire on events (failed deliveries drive outbound calls in phase 7) |
| `swiftroute/db.py` | SQLite schema + deterministic seed data. Portable SQL, so Postgres is a one-function swap |

### Headline result: tracking-ID capture

`python bench/id_capture.py` — 20,000 parcels, 1,500 simulated calls per noise level,
synthetic ASR error model (sound-alike swaps, random swaps, dropped and stuttered chars).

| per-char ASR error | naive exact lookup | resolver top-1 | wrong parcel acted on silently | re-ask needed |
|---|---|---|---|---|
| 3% | 79.3% | **98.1%** | 0.00% | 1.9% |
| 8% | 53.9% | **92.5%** | 0.00% | 7.5% |
| 15% | 30.0% | **78.3%** | 0.00% | 21.7% |

Resolve time p95 ≈ 3.5 ms. The resolver never touches the full table: it generates
plausible mishearings, the checksum discards ~97% of them for free, and one indexed
lookup checks which remaining candidates exist. Cost doesn't grow with the number of parcels.

Phase 5 replaces the synthetic noise with real recordings through faster-whisper and
reuses the same scorer.

### Design decisions worth defending

- **The LLM never approves anything.** Tools return `{allowed, reason_code, explanation}`
  from `policy.py`. The model phrases it; it can't override it.
- **Safety state lives in the session, not the prompt.** A model that hallucinates "the
  caller is verified" or invents a tracking ID still hits the check in `CallSession`.
- **Confirm only what's uncertain.** An exact, checksum-valid match is accepted without a
  confirmation turn (saves ~3 s per call). A corrected match must be confirmed, and the
  read-back starts at the first corrected character, so the caller confirms the part that
  was actually in doubt.
- **Two-factor identity.** Caller ID + spoken postal code = full access. Postal code
  alone = status only. A withheld number counts as unknown: the first version gave
  withheld callers full access, which let anyone skip a factor by hiding caller ID.
  The phase 2 eval transcripts surfaced it.
- **Every tool call writes an audit row with a reason code.** The ops dashboard and call
  tagging (phase 7) are queries over that table, not a second model.

## Phase 2 — text agent + scripted-call evals

| Module | What it does |
|---|---|
| `swiftroute/llm.py` | ~60-line client for any OpenAI-compatible server (Ollama, vLLM, llama.cpp). Model = a flag |
| `swiftroute/agent.py` | Tool loop, voice-style system prompt, context management, per-turn metrics |
| `evals/harness.py` | 14 scenarios + a rule-based caller that knows the ground truth and answers in any order |
| `evals/rule_agent.py` | A no-LLM reference agent: proves the scenarios are passable and runs the harness in CI |

**Context management.** Ground truth (confirmed parcel, verification level, completed
actions, bookable dates) is rebuilt from the session into a `CALL STATE` block on every
LLM call. That makes it safe to keep only the last 6 turns of transcript, so prompt size
stays flat however long the call runs. The state block goes at the *end* of the prompt;
the system prompt and older history stay byte-identical between calls, so servers with
prefix caching reuse the KV cache and only the tail is recomputed.

**Structural guards, not prompt rules.** The model can't confirm a corrected tracking
number in the same turn it was found (`CONFIRM_NEEDS_CALLER`): the caller must answer
first. Hallucinated tools and bad arguments come back as `UNKNOWN_TOOL` / `BAD_ARGS`
and are counted. A loop that doesn't finish in 5 steps hands off to a human instead of
leaving the caller in silence.

**Scenarios** cover the happy paths, a misheard ID, a weekend reschedule, policy refusals,
wrong postal code, social engineering, a prompt injection, and a withheld caller ID.
Each is scored on outcomes, not wording: required reason codes, forbidden reason codes
(safety failures), a DB check, wrong parcel, and "false success claims" (the agent
said "done" in a turn where no action succeeded). The test suite checks the harness
itself: the reference agent passes all 14, and a lying agent and a reckless agent fail.

### Eval log

| run | model | pass | false claims | notes |
|---|---|---|---|---|
| v1 | qwen2.5:7b (M-series Air, Ollama) | 4/14 | 2 | LLM p50 2.5 s / p95 9.3 s, ~1.25k prompt tokens |
| v2 | qwen2.5:7b | 11/14 | 1 | p50 2.4 s / p95 8.7 s; 2 nudges used |
| v2 | qwen2.5:3b | 7/14 | 2 | p50 1.2 s / p95 3.7 s; loops on "okay", parrots tool notes |

What v2 showed, and what changed for v3:
- **The LLM did date arithmetic and got it wrong.** Caller said "Saturday"; 7B booked
  Wednesday the 21st and said "rescheduled for Saturday, October 21". `reschedule_delivery`
  now takes the caller's own words ("Friday", "Monday the 19th") and `dates.py` parses
  them, same principle as tracking IDs: deterministic work stays out of the model.
- **Unbacked success claims still slipped through** ("The address has been changed
  successfully" with no tool call). The guard is now at *runtime*, not just in the eval:
  an unbacked claim gets one correction; if the model repeats it, the reply is replaced
  with a hand-off. A lie never reaches the caller.
- **Promises hidden behind a question** ("I will hold it at the depot. Anything else?")
  dodged the promise check. Checks now run per sentence.
- **Both models parroted tool notes aloud** ("Ask the caller to confirm..."). ID lookup,
  confirmation and postal-code mismatch are deterministic states, so the tools now return
  caller-ready `reply` text and the agent speaks it directly, **skipping an LLM round
  trip** (~1.2–2.4 s each) on those turns.

What v1 showed, and what changed:
- **Status leaked through the state block.** It included the parcel status, so the model
  answered "where's my parcel" without calling `get_status`, a path around verification.
  Status is no longer in the state block.
- **Narrated actions.** "Let me do that for you." with no tool call; a literal `[getstatus]`
  in the reply; "I'll schedule it for Monday" and the turn ends. The agent now gives the
  model one corrective retry per turn when it promises an action or writes a tool name
  instead of calling it, and strips tool names and raw tracking numbers before speech.
- **Ignored "status only" for a partly verified caller** and claimed a reschedule it
  never made (the tools refused it; the eval caught the false claim). The state block now
  carries an explicit `Next:` instruction per verification level.
- **Harness bugs, not model bugs:** the scripted caller answered any *mention* of a postal
  code (so "your postal code matches" got the postal code again, 10 times), hung up on
  "is that correct?", and the success-claim regex flagged "has been verified ... can now
  change". Fixed, with each real Qwen reply kept as a regression test.

| v3 | qwen2.5:7b | **14/14** | 0 | p50 3.0 s / p95 10.1 s per LLM call |
| v3 | qwen2.5:3b | 10/14 | 0 | p50 1.6 s / p95 4.9 s; forgets the caller's goal after verification |

v3 also showed nothing capped postal-code guesses (13 tries in one call). Three failed
attempts now lock verification for the call and hand off to a human.

**The trade-off that drives phase 6:** 7B is correct but ~3 s per LLM call; 3B is twice
as fast but fails 4/14. On a phone line both matter. Plan: distil, i.e. fine-tune the 3B
on the 7B's passing trajectories, then quantise.

## Phase 3 — voice loop (browser)

```
mic -> Silero VAD -> faster-whisper -> smart-turn v3 -> SwiftRouteBrain(Agent) -> Kokoro -> speaker
```

| Piece | Choice | Why |
|---|---|---|
| Transport | Pipecat SmallWebRTC + prebuilt client | browser now; the same bot serves Twilio in phase 4 (`-t twilio`) |
| VAD | Silero, 0.2 s stop | fast "is there sound"; it doesn't decide the turn |
| Turn end | smart-turn v3 (bundled ONNX) | decides the caller is *done*, so a pause while reading an ID doesn't trigger a reply |
| STT | faster-whisper `small.en`, int8 | supports `hotwords` + `initial_prompt`: biased to NATO words and spaced IDs |
| LLM | our `Agent`, not Pipecat's LLM service | keeps our tool loop, guards and fast path, all covered by the evals |
| TTS | Kokoro 82M | open weights, CPU |

`swiftroute/voice/brain.py` is the adapter, and where the voice-specific behaviour lives:
- **Barge-in:** if the caller talks while the agent is still thinking, the stale reply is
  never spoken; if they cut it off mid-sentence, the agent's history is annotated
  ("caller interrupted partway through") so the model doesn't refer to words never heard.
- **Filler:** a turn that runs past 1.2 s says "One moment." so the line is never dead.
- **Chunked TTS:** replies go out sentence by sentence, and long sentences are split at
  commas. Measured with Kokoro on CPU: a 120-character read-back took **3.9 s** to first
  audio as one sentence and **1.7 s** with its first clause sent alone.
- **Latency log:** every turn appends endpointing, STT, agent, TTS and voice-to-voice ms to
  `data/latency.jsonl`; `python -m swiftroute.voice.latency_report` gives p50/p95 per stage.

### Real call #1: what broke, and the fix

First live test (MacBook Air, faster-whisper `small.en`, parcel `SRVCMC7RAJ1`). The
caller read the ID three times and the agent failed all three. Whisper's actual output:

| reading | what Whisper wrote | v1 parser | now |
|---|---|---|---|
| 1 | `S Odd. V C. M C. 7. Hard E G 1.` | not found | recovered, confirmed by read-back |
| 2 | `S. V C. M C. 7 Odd. 1.` | not found | recovered, confirmed by read-back |
| 3 | `S for Snake. R for rat ... M4MAT ... R 4 RAT A4 and... J for Joker 1.` | garbage | **exact** |

- **Spelling by word** ("R for rat", "J for Joker") is how people really spell, and
  Whisper renders the word *for* as the digit **4** ("R 4 RAT", "M4MAT"). The parser now
  treats letter + for/4/like + word as one letter, but keeps "B 4 seven" as B,4,7.
- **Accent homophones:** an "aar" R came out as "odd" / "hard". Added as R.
- **Prefix variants:** "S <R lost> V C ..." is tried as several plausible prefixes, scored
  by checksum + DB, instead of one guess.
- **Use every reading in the call.** Each failed reading is kept; when none resolves alone,
  every parcel is scored against *all* readings with the confusion-weighted distance.
  Independent mishearings rarely hit the same character, so evidence adds up. The real
  readings 1 and 2 each score the right parcel at ~2.0 vs ~4.0 for the next best (20k
  parcels). A C-speed rapidfuzz shortlist keeps that at ~40–60 ms. Only a clear winner is
  offered, and the caller still confirms it.
- **Escalating re-ask:** second failure asks for "a word for each letter, like R for
  Romeo"; third offers a colleague.

All three transcripts are regression tests in `tests/test_real_call_ids.py`.

### Voice latency log

Real calls on a MacBook Air (qwen2.5:3b via Ollama, faster-whisper small.en int8 on CPU, Kokoro):

| run | turn detection | STT | agent | TTS first audio | **voice-to-voice p50 / p95** |
|---|---|---|---|---|---|
| call #1 (19 turns) | 2998 | 1222 | 1312 | 633 | **4669 / 7059 ms** |
| call #3 (10 turns), adaptive endpointing | **1092** | 1092 | 10907 | 810 | 11006 / 26560 ms |
| call #4 (10 turns) | 1327 | 1327 | 5310 | 914 | 7210 / 15505 ms |

Call #4 split the agent time: **1.4 LLM calls per turn, but 3.9 s per call** (prompt ~1.5k
tokens). The same 3B model averages 1.6 s per call in the text-only evals, so per-call cost
roughly doubles when Whisper, Kokoro and the LLM share an 8–16 GB laptop. The remaining
latency is a serving problem (smaller prompt, a GPU box, a distilled model), not a
pipeline problem; that's phase 6.

Call #1 diagnosis: turn detection was *exactly* 3.0 s on every turn. That's Pipecat's
smart-turn silence fallback (`stop_secs=3`), not the model's decision; smart-turn was
answering "incomplete" and every turn waited out the timer. STT ran in parallel, so it
was hidden behind that wait, and becomes the bottleneck as soon as endpointing is fixed.

Fix (`swiftroute/voice/endpointing.py`), using what this agent knows about its own call:
- **The silence fallback follows the question just asked:** 2.2 s while collecting a tracking
  number (people pause between chunks), 1.6 s for a postal code, 0.9 s otherwise.
- **The checksum ends the turn:** if the transcript already parses to a checksum-valid
  tracking number, or a full postal code, the agent starts immediately instead of waiting
  for the timer. A half-read ID can't pass the checksum, so this never cuts anyone off.
- Smart-turn's verdict and probability are logged per turn, so the report shows how often
  the model vs the fallback ended the turn.
- The filler is timed from when the *caller* went quiet (1.8 s), not from when the agent
  started.
- LLM load and library imports are warmed during the greeting (first-turn p95 was 5 s).

### Real call #2: interruptions, and what else it showed

Speech recognition held up (both tracking numbers resolved first time). Barge-in did not:
replies started ending with "[caller interrupted; this reply was NOT heard]" spoken aloud,
three times over.
- **Cause:** the agent wrote interruption notes *into the model's history*, appended to the
  assistant's own words. qwen2.5:3b did what small models do: it copied the pattern into
  its next replies. One barge-in was also counted 2–3 times (Pipecat sends an
  InterruptionFrame alongside UserStartedSpeakingFrame).
- **Fix:** a reply the caller never heard is removed from history (what actually happened
  is in CALL STATE). A reply cut off midway is flagged in the CALL STATE block only,
  never in the transcript. One barge-in per turn, and bracketed text is stripped before TTS.

Also from call #2:
- **Invented facts:** asked "when's the next attempt?" / "what time?", the model made up
  "tomorrow". `get_status` now returns the next delivery date and the delivery window, and
  the prompt says to state only dates and times a tool gave.
- **"Let's check the status." with no tool call** slipped past the promise check. "let's" now counts.
- **Kept chatting after "I'll transfer you".** After a transfer the bot now only holds the line.
- **Filler on almost every turn** ("Just a sec. Let me look."): now only after 2.5 s of
  silence from the caller's side.

### Real call #3: repetition, the wrong parcel, and a slow agent

- **Endpointing fix worked:** turn detection 2998 → **1092 ms** p50.
- **Replies repeated.** The 3B model often writes a full answer *and* calls a tool in the
  same message; that pre-tool text was being spoken as "filler", then the real answer after
  the tool. Now only a short filler with no question in it is spoken from a tool-calling
  message, and no sentence is said twice in one reply.
- **"Another package arrived damaged" was applied to the first parcel.** CALL STATE now
  scopes every action to the confirmed parcel and says to get the new tracking number
  first. New eval scenario `second_parcel_claim` checks the claim lands on parcel B and
  not on A.
- **Invented "additional charges" for same-day delivery.** No-fees rule added.
- **The agent took 11 s p50.** The log only held the total, which can't separate "too many
  LLM calls" from "each call is slow". The voice log now records every call's latency and
  prompt size, and the report breaks them out.

### Real call #5: codes said in pieces, and a model that stopped checking

The caller said the postal code in pieces ("It's V8." … "A1L1"), so it arrived as separate
turns. When it finally came in full, qwen2.5:3b didn't call `verify_caller` at all: it
repeated its earlier "that still doesn't match" line, imitating its own history.

Fix (`swiftroute/slots.py`): **when the agent has just asked for a tracking number or
postal code, the answer is handled in code, not by the LLM.** Fragments are merged across
turns with a natural "Mm-hm, go on.", and a complete code goes straight to the tool. The
LLM only comes back once the caller is verified and something open-ended needs doing.
Utterances that aren't just a code ("I don't have it, I'm the owner") still go to the
model. Every ID and postal-code turn now costs **zero LLM calls**.

### Real call #6: a transfer nobody asked for

The caller corrected their name ("Actually my name is Anmol"). qwen2.5:3b called
`transfer_to_human`, then carried on asking for the tracking number as if it hadn't. From
then on every turn got "You're in the queue for a colleague." Transfers were the one
action the model could take with no check at all.

Fix: a model-initiated transfer is allowed only if the caller asked for a person in their
own words, or said yes to an offered colleague; otherwise it returns
`TRANSFER_NOT_REQUESTED` and the model keeps helping. The code's own hand-offs (verification
lockout, loop limit, an unbacked claim) still go through, and the model can't pass the
`system` flag through tool arguments. A successful transfer speaks its own line, so the
model can't keep chatting after it.

### Where to stop patching

Call #4 also showed the model opening a damage claim on the *first* parcel after the
caller said "I received another package". That's a correctness bug, so it got a
structural guard: once the caller mentions another / a different / a second parcel,
every action tool returns `OTHER_PARCEL_NOT_FOUND` until `find_parcel` runs. Prompt rules
alone didn't stop it.

Everything else from the live calls (odd phrasing, mishearing "parcel" as "purse group",
the occasional over-long answer) is model quality, not pipeline logic. The rule this
project follows: **safety and correctness get a structural guard plus a regression test;
conversational polish gets a better model, measured by the evals**, not one-off patches.

## Phase 4 — a real phone number

Same bot, same pipeline: Pipecat's Twilio serializer handles 8 kHz μ-law audio. Two
additions:
- **`/twiml` route.** Twilio only passes the caller's number into the media stream if the
  TwiML asks for it (as a `<Parameter>`); Pipecat's built-in TwiML doesn't. `/twiml`
  does, so on the phone line **caller ID is a real verification factor**, the same
  two-factor check as in the evals.
- **`SWIFTROUTE_OWNER_PHONE`** links your mobile to the demo customer, so a call from it
  plus the postal code gets full verification.

```bash
ngrok http 7860                                     # copy the https host
SWIFTROUTE_OWNER_PHONE=+1XXXXXXXXXX SWIFTROUTE_MODEL=qwen2.5:3b \
  python -m swiftroute.voice.bot -t twilio -x <host>.ngrok-free.dev
# Twilio console > your number > Voice > "A call comes in": Webhook, POST,
#   https://<host>.ngrok-free.dev/twiml
```

**Outbound: the bot calls you about a missed delivery** (the start of phase 7):

```bash
export TWILIO_ACCOUNT_SID=AC... TWILIO_AUTH_TOKEN=... TWILIO_FROM=+1<your Twilio number>
python -m swiftroute.voice.call_me --to +1<your mobile> --host <host>.ngrok-free.dev
```

It opens with the parcel ("we tried to deliver your parcel today but couldn't, the one
ending in ..."), with the parcel pre-selected, but whoever answers still has to give
the postal code before hearing anything about it. A phone can be answered by anyone.

### Telnyx (works on a trial; Twilio's trial blocked media streams)

Twilio's trial fetched our TwiML but never opened the media stream, inbound or outbound
(it played its own "test call" message instead). Telnyx is wired the same way through
Pipecat (`-t telnyx`) and sends the caller and called numbers in the stream's start
message, so caller ID needs nothing extra. On outbound calls the stream's "from" is *our*
number, so the bot treats the number it called as the customer.

```bash
# .env: TELNYX_API_KEY=KEY...  TELNYX_APP_ID=<TeXML application id>  TELNYX_FROM=+1...
#       SWIFTROUTE_OWNER_PHONE=+1<your mobile>
python -m swiftroute.voice.bot -t telnyx -x <host>.ngrok-free.dev
# Telnyx portal > Voice > TeXML Applications > Voice URL: https://<host>.ngrok-free.dev/texml (POST)
python -m swiftroute.voice.call_me --to +1<your mobile> --host <host>.ngrok-free.dev   # outbound
```

## Phase 6 — make each LLM call cheaper, then distil

Live calls showed the bottleneck: **~3.5 s per LLM call** on a MacBook Air, ~1.2 calls per
turn, with a ~1.5k-token prompt.

### 6a. Prompt size vs prefix caching (`swiftroute/prompting.py`): a negative result, kept

First attempt: offer **only the tools valid in the current call state** with short
descriptions, plus a compact system prompt. Fixed prompt 41–67% smaller per state.

Measured on qwen2.5:3b (Ollama, M-series Air, 15 scenarios × 1):

| mode | prompt tok (mean) | LLM p50 | LLM p95 | pass | bad tool calls |
|---|---|---|---|---|---|
| full (original) | 1488 | 936 ms | 3122 ms | 60% | 3 |
| gated, run 1 | 989 | 1736 ms | 5800 ms | 53% | 3 |
| gated, run 2 | 1061 | 1082 ms | 3620 ms | 60% | 11 |

**~30% fewer tokens, not faster.** Ollama reuses the KV cache for the longest prompt
prefix shared with the previous call, and it renders the tool list at the *start* of the
prompt. With every tool always present, that block is identical on every call and almost
free after the first. Gating changes it on every state change, so the cache is lost exactly
where the big block is. Dropping per-argument descriptions also made the 3B guess argument
names (bad tool calls 3 → 11). Lesson: **on a local server, prompt stability matters
more than prompt length.**

Fix (`--prompt compact`): the short system prompt plus **all** tools, with
short descriptions but argument hints kept. The fixed block is 27% smaller (~320 tokens)
and **byte-identical on every call**, so the cache survives. `gated` is kept as a mode for
the comparison, and as the right choice on servers without prefix caching.

```bash
python evals/compare.py --repeat 3 --prompt full    --model "3B=qwen2.5:3b"
python evals/compare.py --repeat 3 --prompt compact --model "3B=qwen2.5:3b"
python evals/compare.py --repeat 3 --prompt gated   --model "3B=qwen2.5:3b"
```

Result (qwen2.5:3b, 15 scenarios × 3 = 45 calls per row):

| mode | pass | LLM p50 | LLM p95 | prompt tok | false claims | bad tool calls |
|---|---|---|---|---|---|---|
| full | 60% | 1366 ms | 3917 ms | 1490 | 7 | 9 |
| compact | 56% | 1646 ms | 4314 ms | 1276 | 5 | **45** |
| gated | 56% | 1318 ms | 4869 ms | 985 | 8 | 9 |

**Conclusion: prompt engineering is a dead end for the 3B.** Pass rate stays within noise
(±4 pts on 45 runs ≈ 2 conversations) and latency doesn't move, because the cost is the
model's decode time, not prompt reading. The same scenarios fail in every mode
(`status_happy`, `damaged_claim`, `hold_at_depot`, `second_parcel_claim`): a model-capability
ceiling. That is the case for 6b. `full` stays the default for the 3B.

Compact's 45 bad tool calls: with the argument descriptions stripped, the 3B sends
arguments the tools don't take, and every `TypeError` cost an extra LLM round trip. Fix in
`tools.py` dispatch (a robustness fix, kept for every mode): extra arguments are dropped and
reported as `ignored_args`; missing *required* ones still fail. `confirm_parcel` takes the
single pending ID when the model sends `{}`, and `transfer_to_human` no longer requires a
summary. `compare.md` now lists the top bad-call signatures and failure reasons per row.

### 6b. Distil 7B → 1.5B with LoRA on the Mac (MLX, free)
- `train/make_dataset.py` runs the teacher (qwen2.5:7b) through randomized training calls
  and records **every LLM call** (exact prompt + gated tools + reply). Kept only if the
  conversation passes its checks with no false claims or broken promises. A reply that
  needed a corrective nudge is dropped, and the retry is kept with the correction removed,
  so the student learns to get it right first time.
- **No leakage:** training uses different seed databases (different parcels and IDs) and
  paraphrased openings, and three scenario types (`prompt_injection`,
  `second_parcel_claim`, `withheld_caller_id`) are **held out of training entirely**.
  The comparison reports trained-type and held-out-type pass rates separately.
- `train/finetune.sh`: `mlx_lm.lora` with `--mask-prompt` (loss on the reply only),
  grad checkpointing, batch 2 × accumulation 4.
- `train/serve.sh`: `mlx_lm.server` with the adapter, an OpenAI-compatible endpoint with
  tool calls, so the agent and evals use it unchanged.
- `evals/compare.py`: one table, all models, same evals.

```bash
bash train/finetune.sh                                   # data + LoRA
bash train/serve.sh                                      # terminal 2: the student
python evals/compare.py --model "1.5B+LoRA=default_model@http://localhost:8080/v1"
ADAPTER= bash train/serve.sh                             # the student before training
python evals/compare.py --model "1.5B base=default_model@http://localhost:8080/v1"
python evals/compare.py --model "3B=qwen2.5:3b" --model "7B=qwen2.5:7b"
```

## Inspector: every step of a call, visible

```bash
python -m swiftroute.voice.bot                 # then open http://localhost:7860/inspector next to the playground
python -m swiftroute.inspector.simulate        # a synthetic caller through the real pipeline, no mic needed
python -m swiftroute.inspector.build data/traces/<call>.jsonl   # -> dist/swiftroute-inspector.html, plays anywhere
```

Every decision point emits a structured event (`swiftroute/trace.py`). Nothing is emitted
unless a call is being traced, so the evals pay nothing. Events stream to the page over a
websocket on the bot's own server, and each call is saved to `data/traces/<time>_<call>.jsonl`
with a `.wav` of both sides for replay.

| Panel | What it shows | Source |
|---|---|---|
| Spectrogram | 80-bin log-mel of the caller (what Whisper reads) and the agent (as heard, on a playout clock), VAD spans, smart-turn verdicts, end-of-turn marks, barge-ins | `voice/taps.py` |
| Pipeline strip | which stage is active right now and its last output | all events |
| Turn trace | per turn, in order: VAD, smart-turn probability, transcript, slot decision, ID parse and the checksum funnel with ranked candidates, the exact prompt (CALL STATE highlighted), the LLM reply and tool calls, each tool result with its policy verdict and reason code, guards that fired, TTS chunks, a latency waterfall | `agent.py`, `tools.py`, `ids.py`, `policy.py`, `voice/brain.py` |
| Call state | the authoritative session state held in code | `trace.session_snapshot` |
| What it handles | each failure seen in a real call or eval and the mechanism for it, with a jump to where it fires in the loaded call | |
| How it gets better | the offline improvement loop, incident table, eval history, phase 6a | |

**The simulation labels what is real.** VAD, smart-turn, the brain, slots, ID decoding,
tools, policy and guards are always the real code. STT, LLM and TTS use the real models
when they're available (`--stt whisper --llm ollama --tts kokoro`), and otherwise a labelled
stand-in: scripted transcripts, the rule-based reference agent, eSpeak NG. The page shows
the stack for every recording. The caller is the eval harness's rule-based caller with an
eSpeak voice; the demo garbles one character of the first ID reading and says "yes" partway
through the read-back, so the correction and barge-in paths are visible.

**No online learning.** Model weights don't change during a call. The system improves
through the loop on the "How it gets better" tab: failure in a trace → structural fix →
regression test → evals → (phase 6b) distillation.

## MCP server and learning between calls

```bash
python -m swiftroute.mcp_server                      # stdio: add to Claude Desktop / Cursor as an MCP server
python -m swiftroute.mcp_server --http --port 8765   # streamable HTTP at /mcp
python -m swiftroute.learning.agent --target qwen2.5:3b --reflector qwen2.5:7b   # improvement agent over MCP
python -m swiftroute.learning.confusions             # learn real ASR confusions from confirmed read-backs
```

**MCP server** (`swiftroute/mcp_server.py`, 22 tools). The call tools are the phone agent's
own tools, behind the same `CallSession`, policy verdicts, audit table and traces, so any
MCP client is one more front end with identical rules. Caller ID sent over MCP is not
trusted (a client can claim any number), so MCP sessions get status-only access unless the
server runs with `--trust-caller-id`. Ops tools read calls, audit trails, reason-code counts
and what the runtime guards caught. Learning tools run evals and manage lessons.

**Improvement agent** (`swiftroute/learning/agent.py`). An MCP client. The reflector model
(7B) runs a baseline, reads failing scenarios and guard findings, and writes one-line
lessons for the phone agent (3B). It cannot promote them: `evaluate_lesson` scores the eval
suite with and without the lesson for two rounds and keeps it only with more passing runs,
no new safety failures or false claims, and held-out scenario types no worse. Lessons are
filtered before testing (nothing about skipping checks, sharing details, admin, fees), sit
after the rules in the prompt, and can't change what `policy.py` allows. `--driver playbook`
runs the same loop with fixed lesson templates, for CI or machines without a model server.

**Learned ID confusions** (`swiftroute/learning/confusions.py`). A caller confirming a
corrected read-back is a free, labelled ASR error. Pairs seen at least twice get cheaper in
the decoder. The update is kept only if replayed real readings resolve at least as well and
the ID benchmark doesn't drop. Exact matches never depend on these costs, so learning can't
make the agent act on a parcel the caller didn't confirm. In tests, readings with an
unseen cross-group confusion went from 0% to 100% resolved after learning.

None of these change model weights. That's phase 6b (distillation), still the slow loop.

## Multi-agent: a call team, and a crew that improves it (`swiftroute/agents`)

Two groups of agents that talk to each other over a typed message bus (`agents/bus.py`).
Which agent may message which is an **allow-list in code**, every message has a hop limit
(two agents revising each other forever is a real failure), and the whole conversation is
logged and traced. Same rule as the rest of the project: agents advise each other, but only
tools change anything and `policy.py` decides every tool call.

### During a call: the call team (`agents/team.py`)

```
caller ⇄ front_desk (agent.py) ──caller.said──► goal_tracker
              │  ▲                                   ▲
   notes.request│  │notes / Next / correction          │goal.get
              ▼  │                                   │
            supervisor ──────────────────────────────┘
              │ handoff.write (on TRANSFER)
              ▼
            scribe
```

| Agent | Job |
|---|---|
| front desk | the existing phone agent: the only one that speaks to the caller or calls tools |
| goal tracker | remembers what the caller asked for in their own words ("hold it at the depot"), across turns, and whether a tool has handled it yet. Deterministic patterns; an optional small LLM for utterances they miss |
| supervisor | coaches the front desk, escalating: (1) once the caller is verified and the goal is known, it **sets the `Next:` line** ("call hold_at_depot now; they already said..."), because small models follow that line best; (2) it **reviews each final reply before it's spoken** and sends it back once if the front desk asks "how can I help?" when it already knows; (3) if the reply *still* asks "what would you like to do?", the caller hears a yes/no offer instead ("Just to confirm, you'd like me to hold the parcel at the depot?"), and a **yes**, or the caller asking again in their own words, has the front desk run that tool **in code**, the way `slots.py` runs ID and postal-code turns, and the LLM only phrases the result. Never for an address change (it needs details only a conversation collects). Also flags repeated replies and caller frustration, and writes a review after the call |
| scribe | writes the hand-off for the human colleague from the session, not the model's memory |

Why: the eval log's main 3B failure is "forgets the caller's goal after verification". The
sub-agents can't call tools themselves, change the session, or speak. Every tool still goes
through `CallSession` and `policy.py`, so a wrong judgement costs a turn, never a safety check.
Notes are partial-verification aware (a "change" goal for a status-only caller produces
"explain they need to call from the account's number", never "call change_address now").
Prompt notes are stripped before TTS, so a parroted "Supervisor:" or "CORRECTION:" is never spoken.

```bash
python evals/run_text.py --model qwen2.5:3b --team          # evals with the call team
python -m swiftroute chat --team --debug                    # shows the agents' messages per turn
SWIFTROUTE_TEAM=1 python -m swiftroute.voice.bot            # voice; reviews go to data/learning/reviews.jsonl
```

Measured (qwen2.5:3b, Ollama, MacBook Air, 15 scenarios × 2, same harness and seeds):

| run | pass | safety failures | false claims | broken promises | nudges | LLM p50 / p95 |
|---|---|---|---|---|---|---|
| front desk alone | 17/30 (56.7%) | 0 | 0 | 3 | 7 | 1090 / 3134 ms |
| team: coaching only (Next line + reply check), partial run | status still failed: the 3B ignored both | | | | | |
| team: + run in code on repeat | 25/30 (83.3%) | 0 | 0 | 0 | 33 | 1850 / 9522 ms |
| **team: + yes/no offer** | **26/30 (86.7%)** | 0 | 0 | 0 | 22 | 1683 / 4073 ms |

Fixed: `status_happy`, `damaged_claim`, `hold_at_depot` (0/2 → 2/2 each), `address_change_ok`
(0/2 → 2/2), `misheard_id` (0/2 → 1/2). Still failing: `second_parcel_claim` (0/2): after the
second parcel is verified, the 3B says "please wait while I check" or offers to reschedule
instead of asking what to do, so the supervisor's "asks what they want" check doesn't fire.
`address_change_locked` dropped one run (2/2 → 1/2): the 3B looped asking for the address it
had been given. Coaching changes how the 3B phrases replies, so expect some ±1-run noise per
scenario at 2 runs each. The latency rise is mostly corrections (each costs an LLM call); the
yes/no offer cut nudges from 33 to 22.

### Between calls: the improvement crew (`agents/crew.py`)

```
coordinator ─► analyst ─► evaluator      what's failing (evals + supervisor reviews), what was already tried
     ├───────► writer ◄─► critic         draft, critique, revise (up to 3 rounds)
     ├───────► evaluator                 the code gate (learning/lessons.py): the only path to promotion
     └───────► curator ─► evaluator      leave-one-out test of promoted lessons; retire the ones that stopped helping
```

The writer can't reach the evaluator (route not allowed), so no lesson is tested without
passing the critic, and nothing is promoted without passing the gate. The critic rejects
lessons that loosen a rule, name no tool, have no trigger, duplicate a promoted lesson, or
repeat one the gate already rejected. The feedback goes back to the writer, which revises.

**How the crew improves itself, not just the phone agent:**
- **Memory** (`data/learning/crew_memory.json`): every attempt and the gate's verdict. The
  writer sees its rejected drafts and why. After 3 rejected lessons for one scenario the
  analyst stops spending eval minutes on it: that failure needs a structural fix, not a lesson.
- **Curation:** lessons that no longer earn their place are retired, which keeps the prompt
  short and its cached prefix stable (see 6a).
- **Field evidence:** the supervisor's reviews of real calls (`forgot_goal`, ...) feed the analyst.

```bash
python -m swiftroute.agents.crew --driver playbook                      # deterministic, no writer model
python -m swiftroute.agents.crew --target qwen2.5:3b --writer qwen2.5:7b --team --cycles 2 --curate
```

Also over MCP: `run_improvement_crew`, `crew_memory`, `agent_messages`, `call_reviews`.
Tests: `tests/test_agents.py` (routes enforced, hop limit, coaching can't bypass policy, the
critic → writer revision loop, memory skipping, curator retirement, the crew over MCP).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m swiftroute seed                 # 250 parcels into data/swiftroute.db
python -m swiftroute sample               # a few IDs + postal codes to test with
python -m swiftroute demo                 # a scripted call, incl. a misheard ID
python -m swiftroute resolve "S R Z A Q D 7 9 F C J"
python -m swiftroute simulate --ticks 5   # move parcels, print outbound-call candidates
pytest -q
python bench/id_capture.py

# phase 2 (needs Ollama: https://ollama.com)
ollama pull qwen2.5:7b
python -m swiftroute chat --debug         # talk to the agent; shows tool calls + latency
python evals/run_text.py --show-failures  # 14 scripted calls, scored
python evals/run_text.py --model qwen2.5:3b
python evals/run_text.py --history all    # context ablation
python evals/run_text.py --model rule-agent   # no model server needed

# phase 3: voice in the browser (first run downloads Whisper small.en + Kokoro, ~800 MB)
pip install -r requirements-voice.txt
SWIFTROUTE_MODEL=qwen2.5:3b python -m swiftroute.voice.bot    # then open http://localhost:7860
python -m swiftroute.voice.latency_report
```

Use headphones: with laptop speakers the bot can hear itself and "barge in" on its own
voice. The console prints the demo parcel's tracking number and postal code to read out.

## Roadmap

1. ✅ Core domain: DB, simulator, policy engine, ID resolver, tests, ID benchmark
2. ✅ Text agent: Qwen 2.5 via Ollama with tool calling; scripted-call evals
3. ✅ Voice loop in the browser: Pipecat + Silero VAD + smart-turn + faster-whisper + Kokoro; per-stage latency
4. ✅ Phone: Twilio trial number → ngrok → Pipecat, caller ID as a verification factor
5. Real-audio ID capture: recorded readings through faster-whisper with hotwords, same scorer
6. 🔧 Model optimisation: prompt size vs prefix cache (measured); 7B→1.5B LoRA distillation on MLX (tooling done, run pending)
7. Proactive outbound calls on failed deliveries + reason-code / FCR dashboard
8. Synthetic-caller eval harness (200+ calls) + 2-minute demo video
9. ✅ Multi-agent: call team (goal tracker, supervisor, scribe) + self-improving crew (writer ⇄ critic, gate, curator, memory)
