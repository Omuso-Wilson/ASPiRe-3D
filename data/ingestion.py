"""
ASPiRe-3D : data/ingestion.py
===============================================================================
Public data-ingestion interface. Re-exports the validated robust CSV reader and
signal ingestion (auto delimiter/header detection, unit & clock conversion).
===============================================================================
"""
from data.data_ingestion import (ingest_signal, CleanedSignal,
                                  _read_csv_columns)

__all__ = ["ingest_signal", "CleanedSignal", "_read_csv_columns"]
