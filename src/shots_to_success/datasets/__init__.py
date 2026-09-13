"""Dataset adapters and the loader that resolves them by name or import path."""

from __future__ import annotations

from typing import Any

from ..registry import import_class
from .base import Dataset
from .beir import BeirDataset
from .trec_tot import TrecTotDataset
from .wands import WandsDataset

BUILTIN: dict[str, type[Dataset]] = {
    WandsDataset.name: WandsDataset,
    BeirDataset.name: BeirDataset,
    TrecTotDataset.name: TrecTotDataset,
}


def load_dataset(spec: str, data_dir: str = "data", **options: Any) -> Dataset:
    """Resolve ``spec`` to a Dataset instance.

    ``spec`` is either a built-in name (``wands``) or an import path to your
    own subclass (``mypkg.datasets:MyDataset``) -- no repo changes needed.
    """

    if spec in BUILTIN:
        return BUILTIN[spec](data_dir=data_dir, **options)
    cls = import_class(spec, Dataset, "dataset", BUILTIN)
    return cls(data_dir=data_dir, **options)


__all__ = ["BUILTIN", "BeirDataset", "Dataset", "TrecTotDataset", "WandsDataset", "load_dataset"]
