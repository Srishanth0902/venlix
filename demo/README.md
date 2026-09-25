# Demo recording

`venlix_demo.mp4` is a 1:30 narrated walkthrough of the dashboard, and [SCRIPT.md](SCRIPT.md) has the 1:00 presenter script, the scene list and the sample cases. This folder can re-record it.

## How it works

`record_venlix.py` drives the real dashboard in a headed Chromium on a virtual X display, and ffmpeg captures the screen.
Each scene's narration is synthesized locally with [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) and placed at
the time the scene actually started. Captions are burned in below the screen. `recorder.py` holds the reusable parts:
cursor, highlights, cards, TTS, capture, sync and encoding.

## Re-recording

Requirements: Linux, `Xvfb`, `ffmpeg` built with libass, Chromium, and the Python packages in `requirements.txt`.

```bash
pip install -r demo/requirements.txt
mkdir -p ~/.tts && cd ~/.tts
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd -

# A fresh database, so the numbers on screen match the narration (6 cases, 83%, ₹155, 35 min)
VENLIX_DB_PATH=/tmp/venlix_demo.db python -m dashboard &
python demo/record_venlix.py
```

Settings (environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `DASHBOARD_URL` | `http://127.0.0.1:8050` | Dashboard to record |
| `DEMO_CHROME` | `/opt/pw-browsers/chromium` | Chromium executable |
| `DEMO_DISPLAY` | `:99` | Virtual display (started if missing) |
| `KOKORO_DIR` | `~/.tts` | Kokoro model files |
| `DEMO_VOICE` / `DEMO_SPEED` | `af_heart` / `1.0` | Narration voice and speed |
| `DEMO_SECONDS` | `90` | Minimum length; the closing card is held until then |
| `DEMO_WORK_DIR` | `demo/.work` | Raw capture, audio and TTS cache |

With a `GEMINI_API_KEY` configured, the same script records the live-model version. The narration's
"without an API key" line would then need editing in `SEGMENTS`.
