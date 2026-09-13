"""Search adapters and the loader that resolves them by name or import path."""

from __future__ import annotations

from typing import Any

from ..registry import import_class
from .base import SearchAdapter
from .bm25 import Bm25Adapter
from .relevan import RelevanAdapter

BUILTIN: dict[str, type[SearchAdapter]] = {
    Bm25Adapter.name: Bm25Adapter,
    RelevanAdapter.name: RelevanAdapter,
}


def load_adapter(spec: str, **options: Any) -> SearchAdapter:
    """Resolve ``spec`` to a SearchAdapter instance.

    ``spec`` is either a built-in name (``bm25``, ``relevan``) or an import
    path to your own subclass (``mypkg.adapters:MyAdapter``) -- no repo changes
    required.
    """

    if spec in BUILTIN:
        return BUILTIN[spec](**options)
    cls = import_class(spec, SearchAdapter, "adapter", BUILTIN)
    return cls(**options)


__all__ = [
    "BUILTIN",
    "Bm25Adapter",
    "RelevanAdapter",
    "SearchAdapter",
    "load_adapter",
]
