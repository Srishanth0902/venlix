"""Vercel entrypoint: Vercel serves the FastAPI `app` it finds in main.py (run locally with `python -m dashboard`)."""
from dashboard.server import app  # noqa: F401
