import os
import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceEmbeddingCache:
    """Stores precomputed reference embeddings for all Bird images."""

    embeddings_by_bird_id: dict
    model_name: str | None
    version: str


def _cache_path(project_root: str) -> str:
    # store in a project-local folder to avoid polluting static/ uploads/
    cache_dir = os.path.join(project_root, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "reference_embeddings.pkl")


def load_reference_cache(project_root: str, expected_version: str):
    path = _cache_path(project_root)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, ReferenceEmbeddingCache):
            return None
        if obj.version != expected_version:
            return None
        # basic sanity check
        if not isinstance(obj.embeddings_by_bird_id, dict):
            return None
        return obj
    except Exception:
        return None


def save_reference_cache(project_root: str, cache: ReferenceEmbeddingCache):
    path = _cache_path(project_root)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp, path)

