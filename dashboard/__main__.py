"""Start the dashboard:  python -m dashboard  (then open http://127.0.0.1:8050)."""
import os
import logging

import uvicorn
from dotenv import load_dotenv
from delivery_agent.env_utils import env_int, env_str


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host = env_str("DASHBOARD_HOST", "127.0.0.1")
    port = env_int("DASHBOARD_PORT", 8050)
    print(f"Venlix Agent Dashboard -> http://{host}:{port}")
    uvicorn.run("dashboard.server:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
