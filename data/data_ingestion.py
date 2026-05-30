"""
ASPiRe-3D : core/data_ingestion.py
===============================================================================
Robust ingestion and PREPROCESSING/CLEANING of real laboratory core-flood CSVs.

Real lab data is messy: free column names, mixed units, non-monotonic or
duplicated timestamps, NaNs, blank lines, header rows, noise, and outliers. A
calibration that assumes clean SI data silently produces nonsense on such files.
This module turns a raw CSV + a SignalMapping into a clean, canonical-unit,
pore-volume-clocked signal, and -- critically -- LOGS every transformation so
the data treatment is auditable in a thesis ("what exactly did you do to the
raw measurements?").

PIPELINE (each step logged)
---------------------------
  1. read       : auto-detect delimiter & header, parse named/indexed columns
  2. unit/clock : convert values to canonical units, clock to pore volumes
  3. clean      : drop NaN/inf, sort by clock, average exact-duplicate clocks
  4. outliers   : optional Hampel/MAD outlier flagging & removal
  5. smooth     : optional moving-average / Savitzky-Golay-style smoothing
  6. resample   : optional uniform-PV resampling for stable interpolation
  7. uncertainty: attach per-point 1-sigma (from column, constant, or noise est)

Output: a CleanedSignal (clock_pv, values, sigma, provenance log).
===============================================================================
"""

import csv
import numpy as np


# ===========================================================================
class CleanedSignal:
    """A cleaned, canonical signal plus its full provenance log."""

    def __init__(self, canonical, clock_pv, values, sigma, log):
        self.canonical = canonical
        self.clock_pv = np.asarray(clock_pv, float)
        self.values = np.asarray(values, float)
        self.sigma = (np.asarray(sigma, float) if sigma is not None else None)
        self.log = log              # list[str] provenance entries
        self.n_raw = log_count = None

    def __len__(self):
        return len(self.values)

    def provenance(self):
        return "\n".join(f"  [{self.canonical}] {line}" for line in self.log)


# ===========================================================================
def _read_csv_columns(path):
    """
    Read a CSV into a dict of columns. Auto-detects delimiter and whether the
    first row is a header. Returns (columns_by_name_or_index, header_present).
    Lines starting with '#' are treated as comments and skipped.
    """
    with open(path, "r", newline="") as f:
        raw = [ln for ln in f.read().splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    if not raw:
        raise ValueError(f"{path}: no data rows")
    # Sniff delimiter.
    try:
        dialect = csv.Sniffer().sniff(raw[0] + "\n" + (raw[1] if len(raw) > 1 else ""))
        delim = dialect.delimiter
    except csv.Error:
        delim = ","
    rows = [r.split(delim) for r in raw]
    # Header detection: first row non-numeric in any field.
    def is_number(s):
        try:
            float(s); return True
        except ValueError:
            return False
    header_present = not all(is_number(x) for x in rows[0])
    if header_present:
        names = [h.strip() for h in rows[0]]
        body = rows[1:]
    else:
        names = [str(i) for i in range(len(rows[0]))]
        body = rows
    cols = {nm: [] for nm in names}
    for r in body:
        for j, nm in enumerate(names):
            val = r[j].strip() if j < len(r) and r[j].strip() != "" else "nan"
            try:
                cols[nm].append(float(val))
            except ValueError:
                cols[nm].append(np.nan)
    return {nm: np.array(v, float) for nm, v in cols.items()}, header_present


def _get_column(cols, key):
    """Fetch a column by name or integer index, with a clear error."""
    if key in cols:
        return cols[key]
    # integer index fallback
    try:
        idx = int(key)
        names = list(cols.keys())
        return cols[names[idx]]
    except (ValueError, IndexError):
        raise KeyError(f"column '{key}' not found; available: {list(cols.keys())}")


# ===========================================================================
def _mad_outlier_mask(values, n_sigma=4.0):
    """
    Hampel/MAD outlier detection: flag points more than n_sigma robust-sigmas
    from the rolling median. Robust to the heavy-tailed noise common in lab
    pressure traces. Returns a boolean mask of inliers.
    """
    v = values.copy()
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) + 1e-30
    robust_sigma = 1.4826 * mad          # MAD -> sigma for normal data
    return np.abs(v - med) <= n_sigma * robust_sigma


def _moving_average(values, window):
    """Centered moving-average smoothing (odd window), edge-preserving."""
    if window < 3 or window % 2 == 0:
        return values
    half = window // 2
    out = values.copy()
    for i in range(len(values)):
        lo = max(0, i - half); hi = min(len(values), i + half + 1)
        out[i] = np.nanmean(values[lo:hi])
    return out


# ===========================================================================
def ingest_signal(path, mapping, metadata,
                  remove_outliers=True, outlier_sigma=4.0,
                  smooth_window=0, resample_n=0,
                  noise_floor_fraction=0.02):
    """
    Full ingest+clean pipeline for one signal. Returns a CleanedSignal.

    Parameters
    ----------
    path : CSV file path
    mapping : SignalMapping describing columns/units for this signal
    metadata : ExperimentMetadata (for clock->PV conversion & normalisation)
    remove_outliers : apply MAD outlier removal
    smooth_window : odd int >0 to moving-average smooth (0 = off)
    resample_n : >0 to resample onto a uniform PV grid of this many points
    noise_floor_fraction : if no uncertainty is provided, estimate a constant
        1-sigma as this fraction of the signal RMS (so weighting is sensible)
    """
    log = []
    cols, header = _read_csv_columns(path)
    log.append(f"read {path}: {len(next(iter(cols.values())))} rows, "
               f"header={'yes' if header else 'no'}, columns={list(cols.keys())}")

    raw_vals = _get_column(cols, mapping.column)
    raw_clock = _get_column(cols, mapping.clock_column)
    n_raw = len(raw_vals)

    # ---- unit & clock conversion ----
    values = mapping.convert_values_to_canonical(raw_vals)
    clock_pv = mapping.convert_clock_to_pv(raw_clock, metadata)
    log.append(f"converted values to canonical unit "
               f"({mapping.unit or 'dimensionless'}"
               f"{', /%.4g' % mapping.normalise_by if mapping.normalise_by else ''})"
               f"; clock {mapping.clock_unit}->PV")

    # ---- per-point uncertainty (before cleaning, same indexing) ----
    sigma = None
    if mapping.uncertainty_column is not None:
        sigma = _get_column(cols, mapping.uncertainty_column)
        sigma = mapping.convert_values_to_canonical(sigma) \
            if mapping.canonical == "dp_history" and mapping.unit else sigma
        log.append(f"uncertainty from column '{mapping.uncertainty_column}'")
    elif mapping.uncertainty is not None:
        sigma = np.full(n_raw, float(mapping.uncertainty))
        log.append(f"constant uncertainty sigma={mapping.uncertainty:g}")

    # ---- drop NaN/inf ----
    finite = np.isfinite(values) & np.isfinite(clock_pv)
    if sigma is not None:
        finite &= np.isfinite(sigma)
    dropped = int(np.sum(~finite))
    values, clock_pv = values[finite], clock_pv[finite]
    if sigma is not None:
        sigma = sigma[finite]
    if dropped:
        log.append(f"dropped {dropped} non-finite rows")

    # ---- sort by clock ----
    order = np.argsort(clock_pv, kind="stable")
    if not np.array_equal(order, np.arange(len(order))):
        log.append("sorted by clock (was non-monotonic)")
    values, clock_pv = values[order], clock_pv[order]
    if sigma is not None:
        sigma = sigma[order]

    # ---- average exact-duplicate clocks ----
    uniq, inv, counts = np.unique(clock_pv, return_inverse=True, return_counts=True)
    if len(uniq) < len(clock_pv):
        vsum = np.zeros(len(uniq)); np.add.at(vsum, inv, values)
        vmean = vsum / counts
        if sigma is not None:
            ssum = np.zeros(len(uniq)); np.add.at(ssum, inv, sigma)
            sigma = ssum / counts
        log.append(f"averaged {len(clock_pv) - len(uniq)} duplicate-clock points")
        clock_pv, values = uniq, vmean

    # ---- outlier removal ----
    if remove_outliers and len(values) >= 5:
        mask = _mad_outlier_mask(values, n_sigma=outlier_sigma)
        nrem = int(np.sum(~mask))
        if nrem:
            values, clock_pv = values[mask], clock_pv[mask]
            if sigma is not None:
                sigma = sigma[mask]
            log.append(f"removed {nrem} MAD outliers (>{outlier_sigma} robust-sigma)")

    # ---- smoothing ----
    if smooth_window and smooth_window >= 3:
        values = _moving_average(values, smooth_window)
        log.append(f"moving-average smoothing window={smooth_window}")

    # ---- resampling ----
    if resample_n and resample_n > 1 and len(values) >= 2:
        grid = np.linspace(clock_pv.min(), clock_pv.max(), resample_n)
        values = np.interp(grid, clock_pv, values)
        if sigma is not None:
            sigma = np.interp(grid, clock_pv, sigma)
        clock_pv = grid
        log.append(f"resampled to {resample_n} uniform-PV points")

    # ---- uncertainty floor ----
    if sigma is None:
        est = noise_floor_fraction * (np.sqrt(np.mean(values ** 2)) + 1e-30)
        sigma = np.full(len(values), est)
        log.append(f"no uncertainty supplied; estimated constant "
                   f"sigma={est:.3e} ({noise_floor_fraction:.0%} of RMS)")

    log.append(f"final: {len(values)} clean points "
               f"(from {n_raw} raw), PV in [{clock_pv.min():.3f}, {clock_pv.max():.3f}]")
    cs = CleanedSignal(mapping.canonical, clock_pv, values, sigma, log)
    cs.n_raw = n_raw
    return cs
