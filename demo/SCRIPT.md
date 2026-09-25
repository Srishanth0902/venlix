# Venlix demo: scripts and walkthrough

Files in this folder:

- `venlix_demo.mp4`: 1:30 narrated screen recording of the dashboard (1920×1080, burned-in captions)
- `venlix_demo.srt`: the same captions as a subtitle file
- `record_venlix.py` / `recorder.py`: the recorder that produced them (see [README.md](README.md))

## 1. The 1:00 script

For presenting live over the dashboard. About 150 words, a comfortable 60 seconds.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:07 | Overview tab | "Every failed delivery costs a business fuel, money and a customer's trust. Venlix stops those failures before they happen." |
| 0:07–0:18 | Click **Run sample cases**; the live feed fills | "Our XGBoost model scores each delivery's risk. The Venlix agent, built on LangGraph, reads those risk factors and picks the right fix: reassign the driver, message the customer, or escalate to a human." |
| 0:18–0:33 | **Cases**: DEL-DEMO-TRAFFIC, then DEL-TEST-4, then DEL-DEMO-FRAUD | "Here, six risky deliveries were handled in milliseconds. A traffic jam gets a new driver, with no LLM call. A gate-access problem becomes an SMS; the customer asks for tomorrow at 2 PM, and the agent reschedules. Fraud goes straight to human review." |
| 0:33–0:47 | **Copilot**, then **Exception analyzer** | "Operations teams also get a Copilot that answers questions over live data, and an exception analyzer that turns messy driver notes into root causes and fixes." |
| 0:47–1:00 | Back to the KPI row, or **System** | "Eighty-three percent resolved without a human, with time and rupees saved on every case, and it keeps working even without an API key. Venlix: fewer failed deliveries, happier customers." |

## 2. The 1:30 video, scene by scene

Timestamps are from the final recording.

| Time | Scene | Narration |
|---|---|---|
| 0:00 | Title card | This is Venlix, an AI agent that rescues at-risk deliveries before they fail. |
| 0:05 | Overview: **Run sample cases**, live activity feed | Our XGBoost model flags risky deliveries. One click sends them through the LangGraph agent, concurrently. The live feed shows each decision as it happens. |
| 0:15 | KPI row, then charts by failure type and outcome | Six cases, eighty-three percent resolved without a human, one escalated, and ₹155 and thirty-five minutes saved. |
| 0:24 | Cases: DEL-DEMO-TRAFFIC, facts, then agent trace | A traffic jam in heavy rain is classified as a driver delay, so the agent reassigns a nearby idle driver instantly, with no LLM call. |
| 0:35 | Cases: DEL-TEST-4, SMS, reply, parsed intent, trace | For a gate-access problem, it texts the customer, reads the reply, extracts the new slot, tomorrow at 2 PM, and reschedules. Every step is traced. |
| 0:45 | Cases: DEL-DEMO-FRAUD | Fraud signals go straight to a human review queue. |
| 0:49 | **New case**, preset "Wrong address", run | Ops teams can also test their own scenario. Pick a preset, like a wrong address, and run it through the agent live. |
| 0:56 | Copilot: "Which cases did the agent escalate, and why?" | The Operations Copilot answers questions over live data, like which cases were escalated, and why. |
| 1:03 | Exception analyzer: gate-code driver note | The exception analyzer turns a messy driver note into a root cause, a fix, and similar past cases. |
| 1:11 | System: provider chain, recent LLM calls | The System tab shows the provider chain: Gemini, OpenRouter, then a built-in offline engine, which runs this entire demo without an API key. |
| 1:21 | Closing card | Venlix: fewer failed deliveries, lower costs, happier customers. Clone the repo and run it with one command, no API key required. |

## 3. Sample cases shown

All results below are what the agent produced in the recording. It ran on the built-in offline engine, because no API key was configured.

| Case | Input (ML risk factors) | What the agent did | Result |
|---|---|---|---|
| **DEL-DEMO-TRAFFIC**, Neha Kapoor, risk 0.91 | Heavy Traffic Congestion 88, Adverse Weather 54, Driver Reliability 20 | Classified as `driver_delay` (100% of risk weight) and took the deterministic path: reassigned a nearby idle driver with no LLM call | Auto-resolved, 15 min and ₹75 saved |
| **DEL-TEST-4**, Rahul Nair, risk 0.99 | High Gate Wait Time 90, Customer Response Time 47, plus minor factors | Classified as `access_issue` (57%). Drafted an SMS about the gate or visitor pass; the simulated reply was "Can you come tomorrow at 2 PM instead?"; parsed intent: wants reschedule, not declined, slot Tomorrow 2:00 PM | Rescheduled, 5 min and ₹20 saved |
| **DEL-DEMO-FRAUD**, Unknown Recipient, risk 0.97 | Suspicious Order Pattern 92, Drop Distance Anomaly 61 | Classified as `fraud` (93%) and routed straight to escalation | Escalated to the human review queue, no savings counted |
| **Custom case**, preset "Wrong address", Priya Sharma, risk 92 | Low Address Confidence 88, Previous Failed Deliveries 20 | Classified as an address issue. Sent an SMS asking the customer to confirm the address and parsed the reply | Rescheduled to Tomorrow 2:00 PM |
| **Copilot**: "Which cases did the agent escalate, and why?" | The agent's own case store | Answered from data: 7 cases, 6 resolved, 1 escalated (DEL-DEMO-FRAUD, fraud signals, sent to human review) | Answer lists its data source |
| **Exception analyzer**: "Gate code 4821 not working, security guard won't let me in, customer phone off" | Historical exception notes (tier-2 RAG) | Retrieved 3 similar gated-access cases | Root cause: outdated gate PIN from the customer. Fix: SMS for an updated code and a dispatch check of the customer profile. Confidence 80% |

KPIs after the six sample cases: 6 processed, 83% resolved without a human, 1 escalated, ₹155 cost saved (plus ₹90 fuel), 35 minutes saved.
One sample prediction is low risk, so the agent skips it. That is why seven samples give six cases.

## 4. Features covered

| Tab | Feature | Where in the video |
|---|---|---|
| Overview | Batch run of ML-flagged deliveries through the LangGraph agent (concurrent), live WebSocket feed, KPIs, charts | 0:05 to 0:24 |
| Cases | Per-case detail: failure type and confidence, resolution path, SMS conversation, parsed intent, ML risk factors, agent trace with timings | 0:24 to 0:49 |
| Cases, New case | Custom scenarios from presets (gate access, unreachable customer, wrong address, traffic, fraud) | 0:49 |
| Copilot | Chat over live operations data and the agent's own cases, streaming | 0:56 |
| Exception analyzer | RAG over historical driver exceptions: root cause, fix, confidence, similar cases | 1:03 |
| System | Provider chain (Gemini, OpenRouter, offline), LLM call log, latency | 1:11 |
