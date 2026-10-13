# SwiftRoute architecture

SwiftRoute is a phone assistant for a parcel courier. Callers can ask where their parcel is, reschedule a delivery, change the address, hold it at a depot, or report it damaged or missing. Everything runs on one machine, except the phone network.

## Architecture overview

```mermaid
flowchart LR
    subgraph Clients["Clients"]
        direction TB
        Phone["📞 Phone caller"]
        Browser["💻 Browser<br/>call + live viewer"]
        McpC["🤖 MCP clients<br/>improvement agent ·<br/>Claude Desktop · Cursor"]
    end

    subgraph Edge["Telephony"]
        direction TB
        Carrier["Twilio / Telnyx"]
        Tunnel["ngrok tunnel"]
    end

    subgraph Host["SwiftRoute — runs on one machine"]
        direction LR
        subgraph VoiceSrv["Voice server · port 7860"]
            direction TB
            Pipe["Voice pipeline<br/>Silero · Whisper ·<br/>Smart Turn · Kokoro"]
            Viewer["Live call viewer"]
        end
        subgraph Runtime["Agent runtime"]
            direction TB
            Desk["Front-desk agent"]
            Team["Helper agents<br/>goal tracker · supervisor ·<br/>note-taker"]
            Tools["Tools + safety rules"]
            Desk <-->|"message bus"| Team
            Desk --> Tools
        end
        McpS["MCP server<br/>SwiftRoute's tools"]
        Crew["Improvement team<br/>+ practice calls"]
    end

    subgraph ModelSrv["Model server"]
        Ollama["Ollama · port 11434<br/>Qwen 2.5 7B / 3B"]
    end

    subgraph Data["Storage"]
        direction TB
        DB[("Parcel database<br/>+ audit log")]
        Rec[("Call recordings<br/>+ logs")]
        Learn[("Lessons +<br/>learned mishearings")]
    end

    Phone -->|"phone call"| Carrier
    Carrier -->|"audio stream"| Tunnel
    Tunnel <-->|"WebSocket audio"| Pipe
    Browser <-->|"WebRTC audio"| Pipe
    Browser -.->|"WebSocket events"| Viewer

    Pipe <-->|"text in · reply out"| Desk
    Desk <-->|"HTTP · OpenAI API"| Ollama
    Tools -->|"SQL reads + writes"| DB
    Viewer -.->|"replays calls"| Rec

    Desk -->|"logs every step"| Rec
    McpC <-->|"MCP · stdio or HTTP"| McpS
    McpS -->|"same tools, same rules"| Tools
    McpS --> Crew
    Desk -->|"reads lessons<br/>at call start"| Learn
    Crew -->|"lesson writer"| Ollama
    Crew -->|"reads failures ·<br/>saves kept lessons"| Learn

    classDef client fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef edge fill:#e0f2fe,stroke:#0284c7,color:#082f49
    classDef voice fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef agent fill:#ede9fe,stroke:#7c3aed,color:#3b0764
    classDef mcp fill:#fce7f3,stroke:#db2777,color:#500724
    classDef model fill:#ffe4e6,stroke:#e11d48,color:#4c0519
    classDef store fill:#f3f4f6,stroke:#6b7280,color:#111827
    class Phone,Browser,McpC client
    class Carrier,Tunnel edge
    class Pipe,Viewer voice
    class Desk,Team,Tools agent
    class McpS,Crew mcp
    class Ollama model
    class DB,Rec,Learn store
```

### Components

| Component | What it is | Talks to |
|---|---|---|
| **Clients** | People calling by phone or from a browser, plus AI tools that connect over MCP | Telephony, voice server, MCP server |
| **Telephony** | Twilio or Telnyx carry phone calls; ngrok gives the local machine a public address | Voice server, over a WebSocket audio stream |
| **Voice server** (port 7860) | The voice pipeline: **Silero** detects speech, **Whisper** turns it into text, **Smart Turn** decides when the caller has finished, **Kokoro** speaks the reply. It also serves the live call viewer. | Callers (audio), front-desk agent (text) |
| **Front-desk agent** | Runs the conversation. Handles tracking numbers and postal codes in plain code, and asks the AI model for everything else. | Voice server, helper agents, Ollama, tools, storage |
| **Helper agents** | Goal tracker, supervisor and note-taker. They watch each turn and coach the front-desk agent over a message bus. | Front-desk agent only |
| **Tools + safety rules** | Every action (status, reschedule, change address, hold, claim, transfer) goes through here. The rules approve or refuse it, and the AI can't skip them. | Parcel database |
| **Ollama** (port 11434) | Runs the **Qwen 2.5** language model, 7B or 3B. Spoken to over HTTP with the OpenAI API format, so vLLM or llama.cpp also work. | Front-desk agent, improvement team |
| **MCP server** | Exposes SwiftRoute's tools to outside AI clients. Call tools use the same safety rules as a phone call. | MCP clients, tools, improvement team |
| **Improvement team** | Between calls, it reads failed calls, writes short lessons, and keeps one only if the practice calls improve. | Ollama (to write lessons), storage |
| **Storage** | The parcel database with an audit log of every action; call recordings and logs; kept lessons and learned mishearings | — |

### How a call moves through it

1. The caller's audio reaches the voice server, by WebRTC from a browser or by WebSocket from Twilio or Telnyx.
2. Whisper turns the speech into text, and the text goes to the front-desk agent.
3. The agent asks Ollama what to say. If it wants to change something, the tools and safety rules check it first, then update the database.
4. Kokoro speaks the reply back down the same connection.
5. Every step is logged. The live call viewer shows it as it happens and can replay it later.
6. Between calls, the improvement team learns from those logs. Kept lessons are loaded at the start of the next call.

## How the MCP client and server talk

MCP (Model Context Protocol) is a standard way for an AI agent to use another program's tools.

- **The server** is SwiftRoute. It offers tools such as "look up a parcel", "run the practice calls", "propose a lesson" and "test a lesson".
- **The client** is the improvement agent. It starts the server, asks which tools exist, then calls them one at a time.

The client can suggest lessons, but only the server decides whether a lesson is kept.

```mermaid
sequenceDiagram
    autonumber
    participant C as Improvement agent<br/>(MCP client)
    participant S as SwiftRoute<br/>(MCP server)
    participant P as Practice calls
    participant L as Learned lessons

    C->>S: Connect and ask which tools exist
    S-->>C: List of tools

    C->>S: "Run the practice calls"
    S->>P: Run all 15 scenarios
    P-->>S: Score and failed calls
    S-->>C: Score and what went wrong

    Note over C: The language model studies<br/>the failures and writes a lesson

    C->>S: "Propose this lesson"
    C->>S: "Test this lesson"
    S->>P: Run the calls with and without it

    alt Calls improve and nothing breaks
        S->>L: Keep the lesson
        S-->>C: Kept ✅
    else Otherwise
        S-->>C: Rejected ❌ with the reason
    end
```

Other MCP clients, such as Claude Desktop or Cursor, can connect to the same server the same way and use its parcel tools under the same safety rules. Because an MCP client could claim to be anyone's phone number, these clients can only check a parcel's status. They can't change a delivery.

Image copies of both diagrams, for viewers that don't show Mermaid: [architecture.png](architecture.png) and [mcp-flow.png](mcp-flow.png).
