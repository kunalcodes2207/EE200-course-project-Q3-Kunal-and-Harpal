"""
fingerprint.py
--------------
Core audio fingerprinting engine for EE200 Q3(B).

Pipeline (mirrors the report, Q3A):
  1. SPECTROGRAM   : short-time Fourier transform -> time-frequency magnitude map.
  2. CONSTELLATION : keep only local-maximum peaks (the strongest, sparsest points).
  3. HASHING       : pair each anchor peak with nearby peaks ahead of it in time;
                      a hash = (freq1, freq2, delta_t). Two frequencies + a time gap
                      is far more specific than either peak alone, so collisions
                      across unrelated songs are rare.
  4. DB LOOKUP     : every hash maps to a list of (song_id, anchor_time) entries.
  5. SCORING       : for every matching hash, vote for an offset = db_time - query_time.
                      A true match produces a sharp spike in the offset histogram
                      (many hashes agreeing on one offset); chance matches scatter.
"""

from __future__ import annotations

import os
import time
import pickle
import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter

# --------------------------------------------------------------------------
# Tunable parameters (also surfaced in the UI for the "experiment" requirement)
# --------------------------------------------------------------------------
SR = 11025                 # working sample rate (Hz) - plenty for fingerprinting
N_FFT = 4096                # window length (samples) -> ~0.37s @ 11025Hz
HOP = 2048                  # hop length (samples)    -> ~0.19s @ 11025Hz, 50% overlap

PEAK_FREQ_NEIGHBORHOOD = 12   # local-max window half-size, frequency axis (bins)
PEAK_TIME_NEIGHBORHOOD = 12   # local-max window half-size, time axis (frames)
AMP_MIN_DB = -42              # discard peaks this far (dB) below the loudest point

FAN_VALUE = 8                 # how many neighbours each anchor peak pairs with
MIN_TIME_DELTA = 1             # frames
MAX_TIME_DELTA = 90             # frames  (~17s window at hop=2048/SR=11025)

FREQ_BITS = 10               # bits used to encode each (quantized) frequency bin
DT_BITS = 12                  # bits used to encode the time delta

# Two different mp3 encodes of "the same" song are never bit-identical: different
# bitrates/encoders apply slightly different lossy filtering, so a peak that lands
# on FFT bin 137 in one file might land on bin 136 or 138 in another. Hashing the
# *raw* bin makes the fingerprint brittle to that jitter. Quantizing (grouping
# adjacent bins together) before hashing trades a little frequency specificity for
# a lot of robustness to exactly this kind of cross-encoding noise.
FREQ_QUANT = 3                # group this many adjacent FFT bins into one hash bucket

# Bump this whenever hash_peaks()/find_peaks()/load_audio() change in a way that
# would make old cached hashes incompatible with newly-computed ones. The app
# checks this before trusting a cached fingerprint_db.pkl from disk.
SCHEMA_VERSION = 2


@dataclass
class FingerprintResult:
    hashes: list                  # list[(hash_int, anchor_time_frame)]
    peaks: list                   # list[(freq_bin, time_frame)]
    mag_db: np.ndarray            # full spectrogram (freq_bins x time_frames), dB
    freqs: np.ndarray
    times: np.ndarray
    timings: dict = field(default_factory=dict)   # stage -> milliseconds


# --------------------------------------------------------------------------
# Stage 1: audio -> spectrogram
# --------------------------------------------------------------------------
def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Decode any of wav/mp3/flac/ogg/m4a to a mono float32 array at `sr` Hz.

    Two copies of "the same" song from different sites are rarely byte-identical:
    they may have different lead-in/trail silence, different loudness normalization
    (ReplayGain, streaming-site mastering), etc. Trimming silence and peak-normalizing
    here means both copies get fingerprinted from materially the same starting point
    and amplitude scale, which removes two common, easily-avoidable sources of
    mismatch before the signal even reaches the spectrogram stage.
    """
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=40)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1e-6:
        y = y / peak
    return y.astype(np.float32)


def compute_spectrogram(y: np.ndarray, sr: int = SR, n_fft: int = N_FFT, hop: int = HOP):
    freqs, times, Zxx = signal.stft(
        y, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None
    )
    mag = np.abs(Zxx)
    mag_db = 20.0 * np.log10(mag + 1e-8)
    return mag_db, freqs, times


# --------------------------------------------------------------------------
# Stage 2: spectrogram -> constellation (sparse local-maximum peaks)
# --------------------------------------------------------------------------
def find_peaks(mag_db: np.ndarray,
               freq_neighborhood: int = PEAK_FREQ_NEIGHBORHOOD,
               time_neighborhood: int = PEAK_TIME_NEIGHBORHOOD,
               amp_min_db: float = AMP_MIN_DB):
    footprint = np.ones((freq_neighborhood * 2 + 1, time_neighborhood * 2 + 1))
    local_max = maximum_filter(mag_db, footprint=footprint, mode="constant", cval=-np.inf) == mag_db
    loud_enough = mag_db > (mag_db.max() + amp_min_db)
    mask = local_max & loud_enough
    freq_idx, time_idx = np.where(mask)
    peaks = list(zip(freq_idx.tolist(), time_idx.tolist()))
    return peaks


# --------------------------------------------------------------------------
# Stage 3: constellation -> hashes
# --------------------------------------------------------------------------
def hash_peaks(peaks,
               fan_value: int = FAN_VALUE,
               min_dt: int = MIN_TIME_DELTA,
               max_dt: int = MAX_TIME_DELTA,
               freq_bits: int = FREQ_BITS,
               dt_bits: int = DT_BITS,
               freq_quant: int = FREQ_QUANT):
    peaks_sorted = sorted(peaks, key=lambda p: (p[1], p[0]))
    n = len(peaks_sorted)
    freq_mask = (1 << freq_bits) - 1
    dt_mask = (1 << dt_bits) - 1
    hashes = []
    for i in range(n):
        f1, t1 = peaks_sorted[i]
        q1 = f1 // freq_quant
        for j in range(1, fan_value + 1):
            k = i + j
            if k >= n:
                break
            f2, t2 = peaks_sorted[k]
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            q2 = f2 // freq_quant
            h = (q1 & freq_mask) << (freq_bits + dt_bits)
            h |= (q2 & freq_mask) << dt_bits
            h |= (dt & dt_mask)
            hashes.append((h, t1))
    return hashes


# --------------------------------------------------------------------------
# Full single-clip fingerprinting (used for both indexing and querying)
# --------------------------------------------------------------------------
def fingerprint_audio(y: np.ndarray, params: dict | None = None) -> FingerprintResult:
    p = dict(
        sr=SR, n_fft=N_FFT, hop=HOP,
        freq_neighborhood=PEAK_FREQ_NEIGHBORHOOD, time_neighborhood=PEAK_TIME_NEIGHBORHOOD,
        amp_min_db=AMP_MIN_DB, fan_value=FAN_VALUE, min_dt=MIN_TIME_DELTA, max_dt=MAX_TIME_DELTA,
    )
    if params:
        p.update(params)

    timings = {}

    t0 = time.perf_counter()
    mag_db, freqs, times = compute_spectrogram(y, sr=p["sr"], n_fft=p["n_fft"], hop=p["hop"])
    timings["spectrogram"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    peaks = find_peaks(mag_db, p["freq_neighborhood"], p["time_neighborhood"], p["amp_min_db"])
    timings["constellation"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    hashes = hash_peaks(peaks, p["fan_value"], p["min_dt"], p["max_dt"])
    timings["hashing"] = (time.perf_counter() - t0) * 1000

    return FingerprintResult(hashes=hashes, peaks=peaks, mag_db=mag_db,
                              freqs=freqs, times=times, timings=timings)


def fingerprint_file(path: str, params: dict | None = None, sr: int = SR) -> FingerprintResult:
    y = load_audio(path, sr=sr)
    return fingerprint_audio(y, params)


# --------------------------------------------------------------------------
# Database: build / persist / query
# --------------------------------------------------------------------------
class FingerprintDB:
    """In-memory hash -> [(song_id, anchor_time), ...] index, with disk persistence."""

    def __init__(self):
        self.hash_table: dict[int, list[tuple[str, int]]] = {}
        self.song_meta: dict[str, dict] = {}   # song_id -> {name, n_hashes, peaks, n_frames}

    # ---- building -------------------------------------------------------
    def add_song(self, song_id: str, name: str, fp: FingerprintResult, source_path: str | None = None):
        for h, t in fp.hashes:
            self.hash_table.setdefault(h, []).append((song_id, t))
        self.song_meta[song_id] = {
            "name": name,
            "n_hashes": len(fp.hashes),
            "n_peaks": len(fp.peaks),
            "n_frames": fp.mag_db.shape[1],
            "peaks": fp.peaks,            # kept for the "full fingerprint" visualization
            "source_path": source_path,   # original file, used to cut "try a sample" clips
        }

    # ---- persistence ------------------------------------------------------
    def save(self, path: str):
        with open(path, "wb") as fh:
            pickle.dump({
                "schema_version": SCHEMA_VERSION,
                "hash_table": self.hash_table,
                "song_meta": self.song_meta,
            }, fh)

    @classmethod
    def load(cls, path: str) -> "FingerprintDB":
        db = cls()
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Cached DB schema_version={data.get('schema_version')} != current "
                f"SCHEMA_VERSION={SCHEMA_VERSION}; needs re-indexing."
            )
        db.hash_table = data["hash_table"]
        db.song_meta = data["song_meta"]
        return db

    # ---- querying -----------------------------------------------------
    def match(self, query_hashes, top_n: int = 5):
        """Returns (ranked_candidates, offset_histograms).

        ranked_candidates: list of dicts {song_id, name, score, offset}, best first.
        offset_histograms: {song_id: {offset: count}} for diagnostic plotting.
        """
        offset_histograms: dict[str, dict[int, int]] = {}
        for h, qt in query_hashes:
            entries = self.hash_table.get(h)
            if not entries:
                continue
            for song_id, dbt in entries:
                offset = dbt - qt
                d = offset_histograms.setdefault(song_id, {})
                d[offset] = d.get(offset, 0) + 1

        ranked = []
        for song_id, hist in offset_histograms.items():
            best_offset, best_count = max(hist.items(), key=lambda kv: kv[1])
            ranked.append({
                "song_id": song_id,
                "name": self.song_meta[song_id]["name"],
                "score": best_count,
                "offset": best_offset,
            })
        ranked.sort(key=lambda r: -r["score"])
        return ranked[:top_n], offset_histograms

    def folder_signature(self) -> str:
        """Used to decide whether the on-disk DB is stale."""
        return hashlib.md5(repr(sorted(self.song_meta.items())).encode()).hexdigest()


# --------------------------------------------------------------------------
# Helpers for building a DB from a folder of audio files
# --------------------------------------------------------------------------
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")


def folder_audio_files(folder: str):
    files = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(AUDIO_EXTS):
            files.append(os.path.join(folder, fn))
    return files


def folder_fingerprint(folder: str):
    """Signature of a folder's contents (names + sizes + mtimes) for cache invalidation."""
    items = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(AUDIO_EXTS):
            full = os.path.join(folder, fn)
            stat = os.stat(full)
            items.append((fn, stat.st_size, int(stat.st_mtime)))
    return hashlib.md5(repr(items).encode()).hexdigest()


def build_database(folder: str, progress_cb=None) -> FingerprintDB:
    db = FingerprintDB()
    files = folder_audio_files(folder)
    for i, path in enumerate(files):
        fn = os.path.basename(path)
        song_id = os.path.splitext(fn)[0]
        y = load_audio(path)
        fp = fingerprint_audio(y)
        db.add_song(song_id, song_id, fp, source_path=path)
        if progress_cb:
            progress_cb(i + 1, len(files), song_id)
    return db
