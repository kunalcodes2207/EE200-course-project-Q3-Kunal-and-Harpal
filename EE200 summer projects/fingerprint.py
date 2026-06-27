"""
fingerprint.py
--------------
Core audio fingerprinting engine for EE200 Q3(B).
Optimized for high precision and robustness.
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
# TUNED DSP PARAMETERS (High-Accuracy Configuration)
# --------------------------------------------------------------------------
SR = 11025                 
N_FFT = 4096                
HOP = 2048                  

# DECREASED neighborhoods to extract a denser constellation of peaks
PEAK_FREQ_NEIGHBORHOOD = 8    
PEAK_TIME_NEIGHBORHOOD = 8    

# INCREASED sensitivity to capture quieter signature frequencies
AMP_MIN_DB = -50              

# INCREASED fan-out from 5 to 15. This creates massively more hash pairs, 
# drastically improving recognition accuracy even on heavily distorted clips.
FAN_VALUE = 15                 

MIN_TIME_DELTA = 1             
MAX_TIME_DELTA = 80            # Expanded the pairing window 

FREQ_BITS = 10               
DT_BITS = 12                  


@dataclass
class FingerprintResult:
    hashes: list                  
    peaks: list                   
    mag_db: np.ndarray            
    freqs: np.ndarray
    times: np.ndarray
    timings: dict = field(default_factory=dict)   


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def compute_spectrogram(y: np.ndarray, sr: int = SR, n_fft: int = N_FFT, hop: int = HOP):
    freqs, times, Zxx = signal.stft(
        y, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None
    )
    mag = np.abs(Zxx)
    mag_db = 20.0 * np.log10(mag + 1e-8)
    return mag_db, freqs, times


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


def hash_peaks(peaks,
               fan_value: int = FAN_VALUE,
               min_dt: int = MIN_TIME_DELTA,
               max_dt: int = MAX_TIME_DELTA,
               freq_bits: int = FREQ_BITS,
               dt_bits: int = DT_BITS):
    peaks_sorted = sorted(peaks, key=lambda p: (p[1], p[0]))
    n = len(peaks_sorted)
    freq_mask = (1 << freq_bits) - 1
    dt_mask = (1 << dt_bits) - 1
    hashes = []
    for i in range(n):
        f1, t1 = peaks_sorted[i]
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
            h = (f1 & freq_mask) << (freq_bits + dt_bits)
            h |= (f2 & freq_mask) << dt_bits
            h |= (dt & dt_mask)
            hashes.append((h, t1))
    return hashes


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


class FingerprintDB:
    def __init__(self):
        self.hash_table: dict[int, list[tuple[str, int]]] = {}
        self.song_meta: dict[str, dict] = {}   

    def add_song(self, song_id: str, name: str, fp: FingerprintResult):
        for h, t in fp.hashes:
            self.hash_table.setdefault(h, []).append((song_id, t))
        self.song_meta[song_id] = {
            "name": name,
            "n_hashes": len(fp.hashes),
            "n_peaks": len(fp.peaks),
            "n_frames": fp.mag_db.shape[1],
            "peaks": fp.peaks,            
        }

    def save(self, path: str):
        with open(path, "wb") as fh:
            pickle.dump({"hash_table": self.hash_table, "song_meta": self.song_meta}, fh)

    @classmethod
    def load(cls, path: str) -> "FingerprintDB":
        db = cls()
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        db.hash_table = data["hash_table"]
        db.song_meta = data["song_meta"]
        return db

    def match(self, query_hashes, top_n: int = 5):
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
        return hashlib.md5(repr(sorted(self.song_meta.items())).encode()).hexdigest()


AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

def folder_audio_files(folder: str):
    files = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(AUDIO_EXTS):
            files.append(os.path.join(folder, fn))
    return files

def folder_fingerprint(folder: str):
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
        db.add_song(song_id, song_id, fp)
        if progress_cb:
            progress_cb(i + 1, len(files), song_id)
    return db