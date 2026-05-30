"""
ASPiRe-3D : data/preprocessing.py
===============================================================================
Public preprocessing/cleaning interface. The cleaning pipeline (NaN removal,
sorting, duplicate averaging, MAD outlier removal, smoothing, resampling,
uncertainty attachment) is implemented inside ingest_signal and exposed here,
together with the building-of-observed-data and config-driven validation entry
points.
===============================================================================
"""
from data.data_ingestion import (ingest_signal, CleanedSignal,
                                  _mad_outlier_mask, _moving_average)
from data.experimental_validation import (build_observed_from_config,
                                           run_experimental_validation,
                                           make_synthetic_dataset, load_config)

__all__ = ["ingest_signal", "CleanedSignal", "_mad_outlier_mask",
           "_moving_average", "build_observed_from_config",
           "run_experimental_validation", "make_synthetic_dataset",
           "load_config"]
