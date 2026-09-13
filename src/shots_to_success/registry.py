"""Resolving datasets and adapters from a name or an import path."""

from __future__ import annotations

import importlib
from typing import Any, Iterable


def import_class(spec: str, base: type, label: str, known: Iterable[str]) -> Any:
    """Import ``module.path:ClassName`` and verify it subclasses ``base``."""

    if ":" not in spec:
        raise KeyError(
            f"unknown {label} {spec!r}. Use a built-in name "
            f"({', '.join(sorted(known))}) or an import path like "
            f"'mypkg.module:MyClass'."
        )
    module_name, _, class_name = spec.partition(":")
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise KeyError(f"{module_name!r} has no attribute {class_name!r}") from None
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise TypeError(f"{spec} is not a {base.__name__} subclass")
    return cls


__all__ = ["import_class"]
