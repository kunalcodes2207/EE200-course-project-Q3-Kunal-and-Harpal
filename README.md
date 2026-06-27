# EE200 — Audio Fingerprinting (Q3-B)

A Shazam-style audio fingerprinting app: indexes a song library as spectrogram
"constellation" fingerprints, then identifies query clips against it — either
one at a time (Identify tab, with full pipeline visualization) or in bulk
(Batch tab, producing `results.csv`).

## Folder layout (must match exactly)

```
Q3 B/
├── app.py
├── fingerprint.py
├── requirements.txt
├── packages.txt
├── README.md
└── EE200 Project Song Database/      <- the provided .mp3 files, unrenamed
```

`app.py` looks for the database folder named exactly
`EE200 Project Song Database` sitting **next to itself**. Don't rename the
mp3 files — each filename (without extension) is used as the song's ID and is
exactly what gets written into `results.csv` as the `prediction` value.

## How it works (matches the Q3-A report)

1. **Spectrogram** — `scipy.signal.stft` (window = `N_FFT`, hop = `HOP` in
   `fingerprint.py`) turns the waveform into a time-frequency magnitude map.
2. **Constellation** — local maxima of the spectrogram (via a 2-D
   `maximum_filter`) are kept as the sparse "peaks" — the fingerprint's raw
   material.
3. **Hashing** — each peak is paired with a few peaks ahead of it in time
   (`FAN_VALUE`); a hash packs `(freq1, freq2, Δt)` into a single integer.
   Two frequencies plus a time gap is specific enough that unrelated songs
   essentially never collide.
4. **DB lookup** — a dict maps `hash -> [(song_id, anchor_time), ...]`.
5. **Scoring** — every matched hash votes for an offset
   (`database_time - query_time`). A true match produces a sharp spike in
   that offset histogram; chance hits scatter into a flat noise floor. The
   song with the tallest spike wins; the spike height is the "cluster score"
   shown in the UI.

All five stages are timed independently and shown in the pipeline strip in
the Identify tab.

## Run locally

```bash
cd "Q3 B"
pip install -r requirements.txt
# librosa needs ffmpeg on PATH to decode mp3 — install it via your OS package
# manager (brew install ffmpeg / apt install ffmpeg / choco install ffmpeg)
streamlit run app.py
```

The first run indexes every file in `EE200 Project Song Database` once (you'll
see a progress bar) and caches the result to `fingerprint_db.pkl` next to
`app.py`, so later runs start instantly. If you add/remove/replace a song
file, the app notices (it hashes folder contents) and re-indexes automatically.

## Deploy to Streamlit Community Cloud

1. Push this whole `Q3 B` folder (including `EE200 Project Song Database/`)
   to a **public** GitHub repo.
2. Go to https://share.streamlit.io → "New app" → pick the repo/branch and
   set **Main file path** to `app.py` (adjust if `app.py` isn't at the repo
   root — e.g. `Q3 B/app.py`).
3. Streamlit Cloud reads `requirements.txt` (Python deps) and `packages.txt`
   (apt deps — `ffmpeg`/`libsndfile1`, needed for mp3 decoding) automatically.
4. Deploy. First boot will index the library (progress bar) and cache it;
   subsequent reloads are fast as long as the underlying container persists.
5. Copy the public app URL — that's what you submit for Q3-B, alongside the
   link to the GitHub repo (the source code).

## Tuning knobs (for the Q3-A "experiment" writeup)

All in `fingerprint.py`:
- `N_FFT` / `HOP` — STFT window length / hop → time–frequency resolution
  trade-off.
- `PEAK_FREQ_NEIGHBORHOOD` / `PEAK_TIME_NEIGHBORHOOD` / `AMP_MIN_DB` — how
  sparse/strict the constellation is.
- `FAN_VALUE` / `MIN_TIME_DELTA` / `MAX_TIME_DELTA` — how peaks are paired
  into hashes (robustness vs. specificity trade-off).
- `MIN_MATCH_SCORE` in `app.py` — confidence threshold below which a query is
  reported as `none`.
