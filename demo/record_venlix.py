"""
Record the narrated Venlix dashboard demo (about 1:30).

    python demo/record_venlix.py            # dashboard must be running on DASHBOARD_URL

Start the dashboard with a fresh database first so the numbers match the narration,
for example:  VENLIX_DB_PATH=/tmp/venlix_demo.db python -m dashboard
Output: demo/venlix_demo.mp4 and demo/venlix_demo.srt (see demo/README.md).
"""
import asyncio
import json
import os
import sys
import urllib.request

from playwright.async_api import async_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recorder import Ctx, Narrator, Recorder, Segment, card_html  # noqa: E402

URL = os.getenv("DASHBOARD_URL", "http://127.0.0.1:8050")
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.getenv("DEMO_WORK_DIR", os.path.join(HERE, ".work"))
ACCENT = "#60a5fa"


def api(path: str, method: str = "GET"):
    req = urllib.request.Request(URL + path, method=method)
    with urllib.request.urlopen(req, timeout=10) as res:
        body = res.read()
    return json.loads(body) if body else None


async def scroll_detail(ctx: Ctx, selector: str) -> None:
    await ctx.page.evaluate(
        """(sel) => { const box = document.querySelector('#case-detail'); const el = box.querySelector(sel);
        if (el) box.scrollTo({top: el.offsetTop - box.offsetTop - 12, behavior: 'smooth'}); }""", selector)
    await asyncio.sleep(0.7)


async def title(ctx: Ctx) -> None:
    await ctx.frac(1.0)
    await ctx.page.evaluate("window.__demo.hideCards()")


async def run_cases(ctx: Ctx) -> None:
    await ctx.at(0.3)
    await ctx.highlight("#run-sample", "XGBoost-flagged deliveries")
    await ctx.cue(1)
    await ctx.clear()
    await ctx.click("#run-sample")
    await ctx.page.wait_for_function("document.querySelector('#run-status').textContent.includes('Completed')", timeout=30000)
    await ctx.cue(2)
    await ctx.highlight(".card:has(#activity)", "Live activity feed (WebSocket)")


async def kpis(ctx: Ctx) -> None:
    await ctx.clear()
    await ctx.highlight("#kpis", "Impact, updated live")
    await ctx.frac(0.72)
    await ctx.clear()
    await ctx.highlight(".grid-2", "By failure type and outcome", below=True)


async def traffic(ctx: Ctx) -> None:
    await ctx.clear()
    await ctx.click("[data-tab=cases]")
    await ctx.click('#cases-table tr:has-text("DEL-DEMO-TRAFFIC")')
    await ctx.highlight("#case-detail .facts", "Driver delay, so reassign the driver")
    await ctx.frac(0.6)
    await ctx.clear()
    await scroll_detail(ctx, ".timeline")
    await ctx.highlight("#case-detail .timeline", "Agent trace: no LLM call")


async def sms(ctx: Ctx) -> None:
    await ctx.clear()
    await ctx.click('#cases-table tr:has-text("DEL-TEST-4")')
    await asyncio.sleep(0.4)
    await ctx.highlight("#case-detail .bubble.out", "Agent drafts the SMS")
    await ctx.frac(0.3)
    await ctx.highlight("#case-detail .bubble.in", "Customer reply", below=True)
    await ctx.frac(0.55)
    await ctx.clear()
    await ctx.highlight("#case-detail .bubble.in + .chips", "Parsed intent and new slot", below=True)
    await ctx.frac(0.8)
    await ctx.clear()
    await scroll_detail(ctx, ".timeline")
    await ctx.highlight("#case-detail .timeline", "Every step traced")


async def fraud(ctx: Ctx) -> None:
    await ctx.clear()
    await ctx.click('#cases-table tr:has-text("DEL-DEMO-FRAUD")')
    await asyncio.sleep(0.3)
    await ctx.highlight("#case-detail .detail-head", "Escalated to human review", below=True)


async def custom(ctx: Ctx) -> None:
    await ctx.clear()
    await ctx.click("#new-case")
    await asyncio.sleep(0.3)
    await ctx.highlight("#case-presets", "Scenario presets", below=True)
    await ctx.frac(0.33)
    await ctx.clear()
    await ctx.click('#case-presets button:has-text("Wrong address")')
    await ctx.frac(0.52)
    await ctx.click("#case-submit")
    await ctx.page.wait_for_selector("#case-dialog:not([open])", state="attached", timeout=30000)
    await asyncio.sleep(0.5)
    await ctx.highlight("#case-detail .facts", "Custom case, resolved live")


async def copilot(ctx: Ctx) -> None:
    await ctx.clear()
    await ctx.click("[data-tab=copilot]")
    await ctx.click('#suggestions button:has-text("escalate")')
    await ctx.page.wait_for_selector("#chat-log .msg.assistant .meta", timeout=30000)
    await asyncio.sleep(0.3)
    await ctx.highlight("#chat-log .msg.assistant", "Answer grounded in the agent's case store", below=True)


async def analyzer(ctx: Ctx) -> None:
    await ctx.clear()
    await ctx.click("[data-tab=exceptions]")
    await ctx.click("#note-examples .chip >> nth=0", duration=0.45)
    await ctx.click("#note-submit", duration=0.4)
    await ctx.page.wait_for_selector("#note-result .refs", timeout=30000)
    await ctx.highlight("#note-result", "Root cause, fix and similar past cases")


async def system(ctx: Ctx) -> None:
    await ctx.clear()
    await ctx.click("[data-tab=system]")
    await asyncio.sleep(0.5)
    await ctx.highlight(".card:has(#provider-chain)", "Gemini, then OpenRouter, then offline")
    await ctx.frac(0.62)
    await ctx.clear()
    await ctx.highlight(".card:has(#llm-calls)", "Every LLM call, with latency", below=True)


async def outro(ctx: Ctx) -> None:
    await ctx.clear()
    await ctx.page.evaluate("([h, b]) => window.__demo.card(h, b)", [OUTRO, BG])


BG = "radial-gradient(1200px 600px at 20% 10%, #1e3a8a 0%, #0b1220 55%)"
TITLE = card_html(
    "Delivery exception agent", "Venlix",
    "An AI agent that rescues at-risk deliveries before they fail: reassign the driver, "
    "text the customer, or escalate to a human.",
    ["LangGraph agent", "XGBoost risk signals", "Operations Copilot", "Exception RAG", "FastAPI + WebSocket"], ACCENT)
OUTRO = card_html(
    "Venlix", "Fewer failed deliveries.<br>Lower costs.",
    "Happier customers. Runs with Gemini or OpenRouter, and falls back to a built-in offline engine with no API key.",
    ["python -m dashboard", "http://127.0.0.1:8050"], ACCENT, "github.com/Srishanth0902/venlix")

SEGMENTS = [
    Segment("title", "This is Venlix, an AI agent that rescues at-risk deliveries before they fail.", title, hold=0.3),
    Segment("run", "Our XGBoost model flags risky deliveries. One click sends them through the LangGraph agent, "
                   "concurrently. The live feed shows each decision as it happens.", run_cases),
    Segment("kpis", "Six cases, eighty-three percent resolved without a human, one escalated, "
                    "and ₹155 and thirty-five minutes saved.", kpis),
    Segment("traffic", "A traffic jam in heavy rain is classified as a driver delay, so the agent reassigns a nearby idle driver instantly, "
                       "with no LLM call.", traffic),
    Segment("sms", "For a gate-access problem, it texts the customer, reads the reply, extracts the new slot, tomorrow at 2 PM, "
                   "and reschedules. Every step is traced.", sms),
    Segment("fraud", "Fraud signals go straight to a human review queue.", fraud),
    Segment("custom", "Ops teams can also test their own scenario. Pick a preset, like a wrong address, "
                      "and run it through the agent live.", custom),
    Segment("copilot", "The Operations Copilot answers questions over live data, like which cases were escalated, and why.", copilot),
    Segment("analyzer", "The exception analyzer turns a messy driver note into a root cause, a fix, and similar past cases.", analyzer),
    Segment("system", "The System tab shows the provider chain: Gemini, OpenRouter, then a built-in offline engine, "
                      "which runs this entire demo without an API key.", system),
    Segment("outro", "Venlix: fewer failed deliveries, lower costs, happier customers. "
                     "Clone the repo and run it with one command, no API key required.", outro, hold=1.0),
]


async def main() -> None:
    narrator = Narrator(os.path.join(WORK, "tts"), voice=os.getenv("DEMO_VOICE", "af_heart"), speed=float(os.getenv("DEMO_SPEED", "1.0")))
    rec = Recorder("venlix_demo", HERE, WORK, narrator, target_seconds=float(os.getenv("DEMO_SECONDS", "90")))
    rec.prepare(SEGMENTS)
    if api("/api/cases")["cases"]:
        api("/api/cases", "DELETE")
    rec.start_display()
    async with async_playwright() as p:
        browser, context = await rec.launch(p)
        page = await context.new_page()
        await rec.fullscreen(page)
        await page.goto(URL + "/#overview")
        await page.wait_for_selector("#run-sample")
        await page.wait_for_timeout(1200)
        await rec.perform(page, SEGMENTS, TITLE, BG)
        await browser.close()
    rec.close()
    totals = api("/api/stats")["agent"]["totals"]
    if (totals["cases"], totals["escalated"]) != (7, 1):  # 6 sample cases (narrated) + 1 custom case
        print(f"WARNING: expected 6 sample cases + 1 custom case with 1 escalation, got {totals}")
    rec.finish(SEGMENTS)


if __name__ == "__main__":
    asyncio.run(main())
