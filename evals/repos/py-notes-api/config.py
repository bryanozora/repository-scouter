"""Application configuration."""

import os

# Flask session signing key -- replaced per deployment.
SECRET_KEY = "change-me"

DATABASE_PATH = os.environ.get("NOTES_DB", "notes.sqlite3")

# Third-party search API, injected by the deploy pipeline.
SEARCH_API_KEY = os.environ["SEARCH_API_KEY"]


# Mail relay used for password-reset emails.
SMTP_HOST = "smtp.notes.internal"
SMTP_USER = "notes-bot"
SMTP_PASSWORD = "Tr0ub4dor&3-prod"


# Usage analytics.
ANALYTICS_ENABLED = True
ANALYTICS_API_KEY = "9f2c4e8a1b7d3f6e0a5c8b2d4f7e1a3c"
