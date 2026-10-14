"""
Universal Tabular Data Loader supporting CSV and Excel formats with robust cleaning.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Tuple, Dict, Any, Union
import pandas as pd


def clean_tabular_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean messy tabular inputs:
    - Strips leading and trailing whitespace from column names.
    - Safely converts stringified numeric columns (e.g. '$1,200.50', '1,200.50', '15%')
      to floats if the vast majority (>=80%) of non-null values represent numbers.
    """
    df = df.copy()
    # 1. Clean header whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # 2. Type inference on object columns
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            sample = non_null.astype(str).str.strip()
            # If values have currency symbol, comma, or percent, check if they are numeric
            cleaned = sample.str.replace(r'[\$,€,£,%]', '', regex=True).str.replace(',', '').str.strip()
            numeric_converted = pd.to_numeric(cleaned, errors='coerce')
            valid_ratio = numeric_converted.notna().sum() / len(sample)
            if valid_ratio >= 0.8:
                full_cleaned = df[col].astype(str).str.replace(r'[\$,€,£,%]', '', regex=True).str.replace(',', '').str.strip()
                df[col] = pd.to_numeric(full_cleaned, errors='coerce')

    return df


def load_dataset(file_source: Union[str, Path, Any], auto_clean: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
            df = pd.read_csv(file_source, sep=sep, encoding="utf-8", na_values=['N/A', 'NA', 'null', 'None', '', 'nan', 'NaN'])
        except UnicodeDecodeError:
            df = pd.read_csv(file_source, sep=sep, encoding="latin-1", na_values=['N/A', 'NA', 'null', 'None', '', 'nan', 'NaN'])
    else:
        try:
            df = pd.read_csv(file_source, encoding="utf-8", na_values=['N/A', 'NA', 'null', 'None', '', 'nan', 'NaN'])
        except Exception:
            df = pd.read_excel(file_source)

    if df is None:
        raise ValueError(f"Unsupported file format: '{filename}'")

    if auto_clean:
        df = clean_tabular_types(df)

    metadata = {
        "filename": os.path.basename(filename),
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }

    return df, metadata
