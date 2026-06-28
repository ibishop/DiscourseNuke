"""Authenticated Bluesky session with token-based storage.

Flow:
  1. First run: log in with handle + app password (read from .env / env vars).
  2. Serialize the resulting session to a token string and cache it in .session.
  3. Later runs: resume straight from the cached session string — no password
     needed, and the refresh token keeps it alive across runs.

Create an app password at: Bluesky → Settings → Privacy and Security →
App Passwords (NOT your main account password).

Put credentials in a .env file (git-ignored):
    BSKY_HANDLE=you.bsky.social
    BSKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
"""

from __future__ import annotations

import os
from pathlib import Path

from atproto import Client
from dotenv import load_dotenv

SESSION_FILE = Path(".session")


def _save_session(client: Client) -> None:
    SESSION_FILE.write_text(client.export_session_string())


def get_client() -> Client:
    """Return a logged-in atproto Client, reusing a cached session if possible."""
    load_dotenv()
    client = Client()

    # Try to resume from a cached session token first.
    if SESSION_FILE.exists():
        try:
            client.login(session_string=SESSION_FILE.read_text().strip())
            # Session may have been refreshed on resume; persist the latest.
            _save_session(client)
            return client
        except Exception as exc:  # expired/invalid token -> fall back to login
            print(f"Cached session unusable ({exc}); logging in fresh.")

    handle = os.getenv("BSKY_HANDLE")
    app_password = os.getenv("BSKY_APP_PASSWORD")
    if not handle or not app_password:
        raise SystemExit(
            "Missing credentials. Create a .env file with BSKY_HANDLE and "
            "BSKY_APP_PASSWORD (an app password from Bluesky settings)."
        )

    client.login(handle, app_password)
    _save_session(client)
    print(f"Logged in as @{handle}; session cached to {SESSION_FILE}.")
    return client
