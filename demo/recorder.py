"""
Narrated screen-recording toolkit for the demo videos.

A recording script (record_*.py) defines segments. Each segment has caption
text and an async action. The Director runs the actions in a real, headed
Chromium on a virtual X display while ffmpeg captures the screen. Afterwards
the narration (Kokoro TTS) is placed at each segment's actual start time, the
captions are burned in below the screen, and everything is encoded to MP4.

Requirements: Xvfb, ffmpeg (with libass), playwright, kokoro-onnx, soundfile,
and the Kokoro model files (kokoro-v1.0.int8.onnx, voices-v1.0.bin) in
$KOKORO_DIR (default ~/.tts). See demo/README.md.
"""
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

W, H, FPS = 1920, 1080, 30
SCALE = 1.25                      # CSS viewport is 1536 x 864
DISPLAY = os.getenv("DEMO_DISPLAY", ":99")
CHROME = os.getenv("DEMO_CHROME", "/opt/pw-browsers/chromium")
SAMPLE_RATE = 24000

# Final frame: the screen is scaled into a dark canvas with a caption band below it.
SCREEN_W, SCREEN_H, SCREEN_X, SCREEN_Y = 1664, 936, 128, 16
BG = "0x0b1220"

# How to say words that TTS gets wrong. Captions keep the written form.
SPEECH = {
    r"\bKPIs\b": "K P Is", r"\bROI\b": "R O I", r"\bLLM\b": "L L M", r"\bSMS\b": "S M S", r"\bAPI\b": "A P I",
    r"\bXGBoost\b": "X G Boost", r"\bFAISS\b": "face", r"\bLangGraph\b": "Lang Graph", r"\bOpenRouter\b": "Open Router",
    r"\bFounderOS\b": "Founder O S", r"\bRAG\b": "rag", r"\bSaaS\b": "sass", r"₹(\d+)": r"\1 rupees",
    r"\b2 PM\b": "2 P M", r"\bOps\b": "ops",
}


def speech_text(text: str) -> str:
    for pattern, spoken in SPEECH.items():
        text = re.sub(pattern, spoken, text)
    return text


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?:])\s+(?=[A-Z“\"'])", text.strip()) if s.strip()]


# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------
class Narrator:
    def __init__(self, cache_dir: str, voice: str = "af_heart", speed: float = 1.0, gap: float = 0.22) -> None:
        self.cache_dir, self.voice, self.speed, self.gap = cache_dir, voice, speed, gap
        os.makedirs(cache_dir, exist_ok=True)
        self._kokoro = None

    def _engine(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro
            model_dir = os.path.expanduser(os.getenv("KOKORO_DIR", "~/.tts"))
            self._kokoro = Kokoro(os.path.join(model_dir, "kokoro-v1.0.int8.onnx"), os.path.join(model_dir, "voices-v1.0.bin"))
        return self._kokoro

    def _sentence(self, text: str) -> np.ndarray:
        spoken = speech_text(text)
        key = hashlib.sha1(f"{self.voice}|{self.speed}|{spoken}".encode()).hexdigest()[:16]
        path = os.path.join(self.cache_dir, f"{key}.npy")
        if os.path.exists(path):
            return np.load(path)
        samples, sr = self._engine().create(spoken, voice=self.voice, speed=self.speed, lang="en-us")
        assert sr == SAMPLE_RATE, sr
        samples = np.trim_zeros(np.asarray(samples, dtype=np.float32), "fb")
        np.save(path, samples)
        return samples

    def render(self, text: str) -> Dict[str, Any]:
        """Audio for a caption text, plus per-sentence cue times (seconds from segment start)."""
        pieces, cues, t = [], [], 0.0
        silence = np.zeros(int(self.gap * SAMPLE_RATE), dtype=np.float32)
        for sentence in split_sentences(text):
            audio = self._sentence(sentence)
            cues.append((t, t + len(audio) / SAMPLE_RATE, sentence))
            pieces += [audio, silence]
            t += (len(audio) + len(silence)) / SAMPLE_RATE
        audio = np.concatenate(pieces[:-1]) if pieces else np.zeros(0, dtype=np.float32)
        return {"audio": audio, "duration": len(audio) / SAMPLE_RATE, "cues": cues}


# ---------------------------------------------------------------------------
# Browser-side helpers: visible cursor, highlights, callouts, full-screen cards
# ---------------------------------------------------------------------------
INIT_JS = r"""
(() => {
  if (window.__demo) return;
  const css = `
  #__demo_cursor{position:fixed;left:0;top:0;width:30px;height:30px;pointer-events:none;z-index:2147483647;
    transform:translate(-200px,-200px);filter:drop-shadow(0 2px 3px rgba(0,0,0,.4))}
  .__demo_ripple{position:fixed;width:44px;height:44px;margin:-22px 0 0 -22px;border-radius:50%;pointer-events:none;
    z-index:2147483646;border:3px solid rgba(245,158,11,.9);animation:__demo_rip .55s ease-out forwards}
  @keyframes __demo_rip{from{transform:scale(.3);opacity:1}to{transform:scale(1.4);opacity:0}}
  .__demo_hl{position:fixed;pointer-events:none;z-index:2147483640;border:3px solid #f59e0b;border-radius:12px;
    box-shadow:0 0 0 6px rgba(245,158,11,.22),0 0 28px rgba(245,158,11,.35);opacity:0;transition:opacity .3s}
  .__demo_hl.on{opacity:1}
  .__demo_tag{position:absolute;left:-3px;top:-38px;background:#f59e0b;color:#111827;font:600 15px/1 Inter,system-ui,sans-serif;
    padding:8px 11px;border-radius:8px;white-space:nowrap}
  .__demo_tag.below{top:auto;bottom:-38px}
  .__demo_callout{position:fixed;z-index:2147483641;max-width:560px;background:rgba(17,24,39,.94);color:#f9fafb;
    font:500 15px/1.45 Inter,system-ui,sans-serif;padding:14px 16px;border-radius:12px;box-shadow:0 12px 32px rgba(0,0,0,.35);
    border:1px solid rgba(255,255,255,.12);opacity:0;transform:translateY(-6px);transition:opacity .3s,transform .3s}
  .__demo_callout.on{opacity:1;transform:none}
  .__demo_callout b{color:#fbbf24;font-weight:600}
  .__demo_callout code{font:500 13.5px/1.4 ui-monospace,Menlo,monospace;color:#a5f3fc}
  .__demo_callout .row{margin-top:6px}
  .__demo_card{position:fixed;inset:0;z-index:2147483645;display:flex;align-items:center;justify-content:center;
    opacity:1;transition:opacity .6s ease;font-family:Inter,system-ui,sans-serif}
  .__demo_card.off{opacity:0}
  `;
  const style = document.createElement("style");
  style.textContent = css;
  const cursor = document.createElement("div");
  cursor.id = "__demo_cursor";
  cursor.innerHTML = '<svg width="30" height="30" viewBox="0 0 28 28"><path d="M5 2.5 L5 22.5 L10.2 17.6 L13.6 25.2 L17.2 23.7 L13.8 16.2 L21 16.2 Z" fill="#111827" stroke="#ffffff" stroke-width="1.7" stroke-linejoin="round"/></svg>';
  const mount = () => { document.documentElement.append(style, cursor); };
  if (document.body) mount(); else document.addEventListener("DOMContentLoaded", mount);
  document.addEventListener("mousemove", (e) => { cursor.style.transform = `translate(${e.clientX - 5}px, ${e.clientY - 3}px)`; }, true);
  document.addEventListener("mousedown", (e) => {
    const r = document.createElement("div"); r.className = "__demo_ripple";
    r.style.left = e.clientX + "px"; r.style.top = e.clientY + "px";
    document.documentElement.append(r); setTimeout(() => r.remove(), 600);
  }, true);

  const q = (target) => typeof target === "string" ? document.querySelector(target) : target;
  window.__demo = {
    rect(target) { const el = q(target); if (!el) return null; const b = el.getBoundingClientRect(); return {x: b.x, y: b.y, w: b.width, h: b.height}; },
    highlight(target, label, opts = {}) {
      const el = q(target); if (!el) return false;
      const b = el.getBoundingClientRect(), pad = opts.pad ?? 8;
      const box = document.createElement("div"); box.className = "__demo_hl";
      Object.assign(box.style, {left: (b.x - pad) + "px", top: (b.y - pad) + "px", width: (b.width + 2 * pad) + "px", height: (b.height + 2 * pad) + "px"});
      if (label) { const t = document.createElement("div"); t.className = "__demo_tag" + (b.y < 70 || opts.below ? " below" : ""); t.textContent = label; box.append(t); }
      document.documentElement.append(box); requestAnimationFrame(() => box.classList.add("on"));
      return true;
    },
    callout(html, pos) {
      const c = document.createElement("div"); c.className = "__demo_callout"; c.innerHTML = html;
      Object.assign(c.style, pos || {right: "24px", top: "24px"});
      document.documentElement.append(c); requestAnimationFrame(() => c.classList.add("on"));
    },
    clear() {
      document.querySelectorAll(".__demo_hl,.__demo_callout").forEach((n) => { n.classList.remove("on"); setTimeout(() => n.remove(), 320); });
    },
    card(html, bg) {
      const c = document.createElement("div"); c.className = "__demo_card"; c.style.background = bg; c.innerHTML = html;
      document.documentElement.append(c); cursor.style.opacity = "0"; return true;
    },
    hideCards() {
      document.querySelectorAll(".__demo_card").forEach((n) => { n.classList.add("off"); setTimeout(() => n.remove(), 700); });
      cursor.style.opacity = "1";
    },
  };
})();
"""


def card_html(kicker: str, title: str, subtitle: str, chips: List[str], accent: str, footer: str = "") -> str:
    chip_html = "".join(
        f'<span style="border:1px solid rgba(255,255,255,.22);background:rgba(255,255,255,.06);padding:9px 16px;'
        f'border-radius:999px;font-size:17px;color:#e5e7eb">{c}</span>' for c in chips)
    return (
        f'<div style="max-width:1100px;padding:0 60px;text-align:left;color:#f9fafb">'
        f'<div style="font-size:18px;letter-spacing:.14em;text-transform:uppercase;color:{accent};font-weight:700">{kicker}</div>'
        f'<div style="font-size:74px;font-weight:800;line-height:1.05;margin:18px 0 18px;letter-spacing:-.02em">{title}</div>'
        f'<div style="font-size:26px;line-height:1.4;color:#cbd5e1;max-width:900px">{subtitle}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:34px">{chip_html}</div>'
        + (f'<div style="margin-top:40px;font-size:16px;color:#94a3b8">{footer}</div>' if footer else "")
        + "</div>")


# ---------------------------------------------------------------------------
# Recording session
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    id: str
    text: str
    action: Optional[Callable[["Ctx"], Awaitable[None]]] = None
    hold: float = 0.8                # pause after the narration ends
    audio: Dict[str, Any] = field(default_factory=dict)


class Ctx:
    """What a segment action gets: the page, timing helpers and UI helpers."""

    def __init__(self, page, seg: Segment, start: float) -> None:
        self.page, self.seg, self.start = page, seg, start
        self.duration = seg.audio["duration"]

    async def at(self, seconds: float) -> None:
        delay = self.start + seconds - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

    async def frac(self, f: float) -> None:
        await self.at(self.duration * f)

    async def cue(self, n: int) -> None:
        """Wait for the n-th sentence of this segment's narration to start."""
        await self.at(self.seg.audio["cues"][n][0])

    # -- pointer ------------------------------------------------------------
    async def move(self, target: str, duration: float = 0.55) -> Tuple[float, float]:
        box = await self.page.locator(target).first.bounding_box()
        if not box:
            raise RuntimeError(f"not visible: {target}")
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        start = getattr(self.page, "_demo_mouse", (x - 240, y + 160))
        steps = max(8, int(duration * 60))
        for i in range(1, steps + 1):
            t = i / steps
            e = t * t * (3 - 2 * t)
            await self.page.mouse.move(start[0] + (x - start[0]) * e, start[1] + (y - start[1]) * e)
            await asyncio.sleep(duration / steps)
        self.page._demo_mouse = (x, y)
        return x, y

    async def click(self, target: str, duration: float = 0.55, settle: float = 0.15) -> None:
        x, y = await self.move(target, duration)
        await asyncio.sleep(0.12)
        await self.page.mouse.click(x, y)
        await asyncio.sleep(settle)

    async def type(self, target: str, text: str, delay: float = 0.028) -> None:
        await self.click(target)
        await self.page.keyboard.type(text, delay=int(delay * 1000))

    async def highlight(self, target: str, label: str = "", **opts: Any) -> None:
        # Resolve with Playwright so selectors like :has-text() work.
        element = await self.page.locator(target).first.element_handle(timeout=10000)
        await self.page.evaluate("([t, l, o]) => window.__demo.highlight(t, l, o)", [element, label, opts])

    async def callout(self, html: str, pos: Optional[Dict[str, str]] = None) -> None:
        await self.page.evaluate("([h, p]) => window.__demo.callout(h, p)", [html, pos])

    async def clear(self) -> None:
        await self.page.evaluate("window.__demo.clear()")
        await asyncio.sleep(0.3)


class Recorder:
    def __init__(self, name: str, out_dir: str, work_dir: str, narrator: Narrator, target_seconds: float = 90.0) -> None:
        self.name, self.out_dir, self.work_dir, self.narrator = name, out_dir, work_dir, narrator
        self.target = target_seconds
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        self.raw = os.path.join(work_dir, f"{name}_raw.mkv")
        self.marks: List[Tuple[str, float]] = []
        self._procs: List[subprocess.Popen] = []

    # -- infrastructure -------------------------------------------------------
    def start_display(self) -> None:
        sock = f"/tmp/.X11-unix/X{DISPLAY.lstrip(':')}"
        if not os.path.exists(sock):
            self._procs.append(subprocess.Popen(["Xvfb", DISPLAY, "-screen", "0", f"{W}x{H}x24", "-nolisten", "tcp"],
                                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            for _ in range(50):
                if os.path.exists(sock):
                    break
                time.sleep(0.1)
        os.environ["DISPLAY"] = DISPLAY

    def start_capture(self) -> None:
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-thread_queue_size", "1024", "-framerate", str(FPS),
             "-video_size", f"{W}x{H}", "-draw_mouse", "0", "-i", f"{DISPLAY}.0",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", self.raw],
            stdin=subprocess.PIPE)

    def stop_capture(self) -> None:
        try:
            self.ffmpeg.communicate(b"q", timeout=30)
        except subprocess.TimeoutExpired:
            self.ffmpeg.kill()

    def close(self) -> None:
        for p in self._procs:
            p.terminate()

    async def launch(self, playwright):
        browser = await playwright.chromium.launch(
            executable_path=CHROME, headless=False, ignore_default_args=["--enable-automation"],
            # --window-size is in scaled (CSS) pixels, so divide by the scale factor to fill the screen exactly.
            args=[f"--window-size={int(W / SCALE)},{int(H / SCALE)}", "--window-position=0,0",
                  f"--force-device-scale-factor={SCALE}", "--no-first-run", "--disable-infobars",
                  "--disable-features=Translate", "--hide-scrollbars", "--disable-background-networking",
                  "--disable-component-update", "--disable-sync", "--no-default-browser-check"])
        context = await browser.new_context(no_viewport=True)
        await context.add_init_script(INIT_JS)
        return browser, context

    @staticmethod
    async def fullscreen(page) -> None:
        """Hide the tab strip and address bar (browser-level fullscreen, not the page's)."""
        cdp = await page.context.new_cdp_session(page)
        window = await cdp.send("Browser.getWindowForTarget")
        await cdp.send("Browser.setWindowBounds", {"windowId": window["windowId"], "bounds": {"windowState": "fullscreen"}})
        await asyncio.sleep(5.0)  # let the "press Esc to exit full screen" bubble disappear

    # -- the run --------------------------------------------------------------
    def prepare(self, segments: List[Segment]) -> None:
        for seg in segments:
            seg.audio = self.narrator.render(seg.text)
        total = sum(s.audio["duration"] + s.hold for s in segments)
        print(f"[{self.name}] narration {total:.1f}s over {len(segments)} segments (target {self.target:.0f}s)")

    async def perform(self, page, segments: List[Segment], title_html: str, title_bg: str) -> None:
        """Show the title card (the sync point), then run every segment in order."""
        self.start_capture()
        try:
            await asyncio.sleep(1.5)
            await page.evaluate("([h, b]) => window.__demo.card(h, b)", [title_html, title_bg])
            self.t0 = time.time()
            for i, seg in enumerate(segments):
                start = time.time()
                self.marks.append((seg.id, start - self.t0))
                ctx = Ctx(page, seg, start)
                if seg.action:
                    await seg.action(ctx)
                overrun = time.time() - (start + seg.audio["duration"] + seg.hold)
                if overrun > 0:
                    print(f"  segment {seg.id!r}: actions ran {overrun:.1f}s past its narration")
                end = start + seg.audio["duration"] + seg.hold
                if i == len(segments) - 1:
                    end = max(end, self.t0 + self.target)
                await asyncio.sleep(max(0.0, end - time.time()))
            self.total = time.time() - self.t0
            await asyncio.sleep(0.4)
        finally:
            self.stop_capture()

    # -- post-production ------------------------------------------------------
    def _sync_offset(self) -> float:
        """Time in the raw capture where the dark title card first appears."""
        cmd = ["ffmpeg", "-loglevel", "error", "-t", "6", "-i", self.raw, "-vf", "scale=64:36,format=gray",
               "-f", "rawvideo", "-"]
        data = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.uint8)
        frames = data.reshape(-1, 36 * 64).mean(axis=1)
        dark = np.nonzero(frames < frames[0] - 60)[0]
        if not len(dark):
            raise RuntimeError("could not find the title card in the capture")
        return dark[0] / FPS

    def _audio(self, segments: List[Segment], path: str) -> None:
        import soundfile as sf
        track = np.zeros(int((self.total + 1) * SAMPLE_RATE), dtype=np.float32)
        for (sid, start), seg in zip(self.marks, segments):
            i = int(start * SAMPLE_RATE)
            audio = seg.audio["audio"][: max(0, len(track) - i)]
            track[i:i + len(audio)] += audio
        peak = float(np.abs(track).max()) or 1.0
        sf.write(path, track * (0.89 / peak), SAMPLE_RATE, subtype="PCM_16")

    def _cues(self, segments: List[Segment]) -> List[Tuple[float, float, str]]:
        cues = []
        for (sid, start), seg in zip(self.marks, segments):
            for a, b, text in seg.audio["cues"]:
                cues.append((start + a, start + b + 0.25, text))
        return cues

    @staticmethod
    def _ts(t: float, sep: str = ",") -> str:
        h, rem = divmod(max(0.0, t), 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s % 1) * 1000)) % 1000:03d}"

    def _write_subs(self, cues, srt_path: str, ass_path: str) -> None:
        with open(srt_path, "w", encoding="utf-8") as fh:
            for i, (a, b, text) in enumerate(cues, 1):
                fh.write(f"{i}\n{self._ts(a)} --> {self._ts(b)}\n{text}\n\n")
        cs = lambda t: f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"
        band_v = H - (SCREEN_Y + SCREEN_H)
        with open(ass_path, "w", encoding="utf-8") as fh:
            fh.write(
                "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n\n"
                "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
                "MarginL, MarginR, MarginV, Encoding\n"
                f"Style: Cap,Inter,36,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,170,170,{max(12, band_v // 2 - 34)},1\n\n"
                "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            for a, b, text in cues:
                fh.write(f"Dialogue: 0,{cs(a)},{cs(b)},Cap,,0,0,0,,{text}\n")

    def finish(self, segments: List[Segment]) -> Dict[str, Any]:
        offset = self._sync_offset()
        wav = os.path.join(self.work_dir, f"{self.name}.wav")
        ass = os.path.join(self.work_dir, f"{self.name}.ass")
        srt = os.path.join(self.out_dir, f"{self.name}.srt")
        out = os.path.join(self.out_dir, f"{self.name}.mp4")
        self._audio(segments, wav)
        cues = self._cues(segments)
        self._write_subs(cues, srt, ass)
        duration = self.total
        vf = (f"color=c={BG}:s={W}x{H}:r={FPS}:d={duration:.3f}[bg];"
              f"[0:v]trim=start={offset:.3f}:duration={duration:.3f},setpts=PTS-STARTPTS,"
              f"scale={SCREEN_W}:{SCREEN_H}:flags=lanczos[scr];"
              f"[bg][scr]overlay={SCREEN_X}:{SCREEN_Y}:shortest=1,"
              f"drawbox=x={SCREEN_X - 2}:y={SCREEN_Y - 2}:w={SCREEN_W + 4}:h={SCREEN_H + 4}:color=0x334155:t=2,"
              f"subtitles={ass},fade=t=in:st=0:d=0.4,fade=t=out:st={duration - 0.6:.3f}:d=0.6[v];"
              f"[1:a]atrim=0:{duration:.3f},afade=t=out:st={duration - 0.6:.3f}:d=0.6[a]")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.raw, "-i", wav, "-filter_complex", vf,
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "21",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], check=True)
        timeline = [{"segment": sid, "start": round(t, 2), "text": seg.text} for (sid, t), seg in zip(self.marks, segments)]
        with open(os.path.join(self.work_dir, f"{self.name}_timeline.json"), "w") as fh:
            json.dump({"duration": round(duration, 2), "sync_offset": offset, "segments": timeline}, fh, indent=2)
        print(f"[{self.name}] wrote {out} ({duration:.1f}s)")
        return {"video": out, "srt": srt, "duration": duration, "timeline": timeline}
