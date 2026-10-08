"""
In-memory store for trained model bundles.

Mirrors session_store.py's design and its tradeoffs: keyed by a UUID
model_id, held in a plain dict, no database. Each successful /api/train call
gets its own model_id (rather than overwriting whatever was last trained for
a session_id), so a user can train several algorithms against the same
uploaded dataset and export any of them afterwards, not just the most recent.

NOTE: resets if the server restarts, doesn't scale across multiple
server processes/workers. Same portfolio-project caveat as session_store -
swap for Redis/S3/a real model registry if this ever needs to survive
restarts or serve concurrent workers.
"""

import uuid
from typing import Dict, Any


class ModelStore:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def create(self, bundle: Dict[str, Any]) -> str:
        model_id = str(uuid.uuid4())
        self._store[model_id] = bundle
        return model_id

    def get(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self._store:
            raise KeyError(f"No trained model found for model_id={model_id}")
        return self._store[model_id]

    def exists(self, model_id: str) -> bool:
        return model_id in self._store


# Single shared instance used across the app
model_store = ModelStore()
