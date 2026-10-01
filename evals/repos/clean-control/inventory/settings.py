"""Settings, all read from the environment."""

import os

DATABASE_URL = os.environ["DATABASE_URL"]


API_TOKEN = os.getenv("INVENTORY_API_TOKEN")


# Default only used by the local docker-compose setup.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")


SESSION_SECRET = "replace-me-in-production"
