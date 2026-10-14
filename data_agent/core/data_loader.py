"""
Universal Tabular Data Loader supporting CSV and Excel formats.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Union
import pandas as pd


def load_dataset(file_source: Union[str, Path, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a dataset from file path or file-like buffer (e.g. Streamlit UploadedFile).
    Returns (DataFrame, metadata_dict).
    """
    filename = getattr(file_source, "name", str(file_source))
    ext = os.path.splitext(filename)[1].lower()

    df = None
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_source)
    elif ext in [".csv", ".tsv", ".txt"]:
        sep = "\t" if ext == ".tsv" else ","
        try:
            df = pd.read_csv(file_source, sep=sep, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_source, sep=sep, encoding="latin-1")
    else:
        # Default try read_csv
        try:
            df = pd.read_csv(file_source, encoding="utf-8")
        except Exception:
            df = pd.read_excel(file_source)

    if df is None:
        raise ValueError(f"Unsupported file format: '{filename}'")

    metadata = {
        "filename": os.path.basename(filename),
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }

    return df, metadata
