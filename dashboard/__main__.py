"""Start the dashboard:  python -m dashboard  (then open http://127.0.0.1:8050)."""
import os
import logging

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.getenv("DASHBOARD_PORT", "8050"))
    print(f"Venlix Agent Dashboard -> http://{host}:{port}")
    uvicorn.run("dashboard.server:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
