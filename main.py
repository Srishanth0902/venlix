"""Vercel entrypoint: Vercel serves the FastAPI `app` it finds in main.py (run locally with `python -m dashboard`)."""
try:
    from dashboard.server import app  # noqa: F401
except Exception:  # only when the deployment is broken: show why instead of an opaque 500
    import sys
    import traceback

    STARTUP_ERROR = traceback.format_exc()
    print(STARTUP_ERROR, file=sys.stderr)

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        body = f"Venlix could not start.\n\n{STARTUP_ERROR}".encode()
        await send({"type": "http.response.start", "status": 500,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8")]})
        await send({"type": "http.response.body", "body": body})
