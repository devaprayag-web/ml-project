"""
In-memory session store.

For an MVP, we don't need a database — each uploaded CSV is held in memory,
keyed by a UUID session_id that the frontend passes back on /train calls.

NOTE: This resets if the server restarts, and doesn't scale across multiple
server processes/workers. That's fine for a portfolio project running as a
single process. If you ever productionize this, swap this dict for Redis.
"""

import uuid
import pandas as pd
from typing import Dict


class SessionStore:
    def __init__(self):
        self._store: Dict[str, pd.DataFrame] = {}

    def create(self, df: pd.DataFrame) -> str:
        session_id = str(uuid.uuid4())
        self._store[session_id] = df
        return session_id

    def get(self, session_id: str) -> pd.DataFrame:
        if session_id not in self._store:
            raise KeyError(f"No dataset found for session_id={session_id}")
        return self._store[session_id]

    def exists(self, session_id: str) -> bool:
        return session_id in self._store


# Single shared instance used across the app
session_store = SessionStore()
