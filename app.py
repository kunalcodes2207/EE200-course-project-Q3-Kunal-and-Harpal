"""
EE200 — Audio Fingerprinting (Q3-B)
Streamlit app: index a song library as spectrogram fingerprints, then identify
short query clips against it (single-clip mode) or in bulk (batch mode).

Folder layout expected (this file lives at the same level as the song folder):

    Q3 B/
      app.py                      <- this file
      fingerprint.py
      requirements.txt
      packages.txt
      EE200 Project Song Database/   <- provided .mp3 files (left untouched / unrenamed)
"""

import os
import io
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fingerprint as fp

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SONG_DB_FOLDER = os.path.join(APP_DIR, "EE200 Project Song Database")
DB_CACHE_PATH = os.path.join(APP_DIR, "fingerprint_db.pkl")
SAMPLES_FOLDER = os.path.join(APP_DIR, "samples")     # optional, only used if present

MIN_MATCH_SCORE = 8     # hashes-on-a-single-offset below this -> "no confident match"

TEAL = "#2dd4c0"
ORANGE = "#f0a868"
BG = "#0a0e0e"
CARD_BG = "#11171710"
PALETTE = ["#5fd0e8", "#e8c75f", "#b08bdb", "#e87aa0", "#7ee0a8", "#e89a5f"]

# --------------------------------------------------------------------------
# Page config + CSS (dark / teal theme to match the project's design language)
# --------------------------------------------------------------------------
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
        color: #d7dede;
    }}
    html, body, [class*="css"] {{
        font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
    }}
    .eyebrow {{
        color: {TEAL};
        letter-spacing: 0.18em;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .title-row {{ display:flex; align-items:center; gap:0.6rem; }}
    .title-text {{ font-size: 2.1rem; font-weight: 800; color: #f1f4f4; }}
    .title-text span {{ color: {TEAL}; }}
    .subtitle {{ color: #6f7a7a; font-size: 0.78rem; letter-spacing: 0.12em; margin-top: -6px;}}
    .desc {{ color: #9aa6a6; font-size: 0.92rem; margin-top: 4px; margin-bottom: 1.2rem;}}

    div[data-testid="stTabs"] button[role="tab"] {{
        font-family: monospace;
        letter-spacing: 0.08em;
        font-size: 0.8rem;
        color: #8a9595;
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {TEAL} !important;
    }}

    .card {{
        background: #10171680;
        border: 1px solid #1d2828;
        border-radius: 10px;
        padding: 14px 16px;
    }}
    .stat-box {{
        background: #0e1515;
        border: 1px solid #1d2828;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }}
    .stat-box .num {{ font-size: 1.3rem; font-weight: 700; color: #eef2f2; }}
    .stat-box .lbl {{ font-size: 0.62rem; letter-spacing: 0.1em; color: {TEAL}; text-transform: uppercase;}}
    .stat-box .sub {{ font-size: 0.65rem; color: #57625f; }}

    .match-panel {{
        background: linear-gradient(180deg, #0f2622aa, #0c1414cc);
        border: 1px solid #1f4a40;
        border-radius: 12px;
        padding: 22px 26px;
    }}
    .match-panel .lbl {{ color: {TEAL}; font-size: 0.72rem; letter-spacing: 0.18em; }}
    .match-panel .name {{ color: #f3f6f6; font-size: 2rem; font-weight: 800; margin: 6px 0 4px 0; }}
    .match-panel .sub {{ color: #8a9a97; font-size: 0.85rem; }}
    .match-panel .sub b {{ color: {ORANGE}; }}

    .step-bar {{ border-left: 3px solid {TEAL}; padding-left: 12px; margin: 18px 0 8px 0; }}
    .step-eyebrow {{ color: {TEAL}; font-size: 0.68rem; letter-spacing: 0.15em; text-transform:uppercase;}}
    .step-title {{ color: #eef2f2; font-size: 1.25rem; font-weight: 700; margin-top: 2px;}}
    .step-desc {{ color: #93a09d; font-size: 0.88rem; margin-top: 4px; }}
    .step-desc b {{ color: {TEAL}; }}

    .song-name {{ color: #e7ecec; font-size: 0.88rem; font-weight: 600; }}
    .song-hashes {{ color: {TEAL}; font-size: 0.74rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="title-row">
    <div class="title-text">EE<span>200</span>: Audio Fingerprinting</div>
</div>
<div class="subtitle">SIGNALS, SYSTEMS &amp; NETWORKS · PROJECT DEMO</div>
<div class="desc">Index a library of songs as spectrogram fingerprints, then identify any short clip against it.</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Database loading (cached; only rebuilt when the song folder actually changes)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_database(folder_signature: str):
    if os.path.exists(DB_CACHE_PATH):
        try:
            db = fp.FingerprintDB.load(DB_CACHE_PATH)
            if db.folder_signature() and len(db.song_meta) > 0:
                return db
        except Exception:
            pass

    db = fp.FingerprintDB()
    files = fp.folder_audio_files(SONG_DB_FOLDER)
    progress = st.progress(0.0, text="Indexing song library…")
    for i, path in enumerate(files):
        fn = os.path.basename(path)
        song_id = os.path.splitext(fn)[0]
        y = fp.load_audio(path)
        fpr = fp.fingerprint_audio(y)
        db.add_song(song_id, song_id, fpr, source_path=path)
        progress.progress((i + 1) / max(len(files), 1), text=f"Indexing… {fn}")
    progress.empty()
    try:
        db.save(DB_CACHE_PATH)
    except Exception:
        pass
    return db


def load_database():
    if not os.path.isdir(SONG_DB_FOLDER):
        st.error(f"Song database folder not found:\n`{SONG_DB_FOLDER}`\n\n"
                 "Place the `EE200 Project Song Database` folder next to app.py.")
        st.stop()
    sig = fp.folder_fingerprint(SONG_DB_FOLDER)
    return get_database(sig)


@st.cache_resource(show_spinner=False)
def get_or_build_samples(folder_signature: str, n_samples: int = 5, clip_seconds: float = 15.0):
    """Cuts a short clip from `n_samples` of the indexed songs so 'OR TRY A SAMPLE'
    always has something to play, with zero manual setup. Clips are written once
    to SAMPLES_FOLDER (as wav) and reused on later runs."""
    db = load_database()
    os.makedirs(SAMPLES_FOLDER, exist_ok=True)

    song_ids = list(db.song_meta.keys())
    step = max(1, len(song_ids) // max(n_samples, 1))
    picked = song_ids[::step][:n_samples]

    samples = []
    for i, song_id in enumerate(picked, start=1):
        out_path = os.path.join(SAMPLES_FOLDER, f"sample{i}.wav")
        meta = db.song_meta[song_id]
        src = meta.get("source_path")
        if not os.path.exists(out_path) and src and os.path.exists(src):
            import librosa
            import soundfile as sf
            full_dur = meta["n_frames"] * fp.HOP / fp.SR
            offset = max(0.0, full_dur * 0.25)   # skip past any intro silence/fade-in
            y, _ = librosa.load(src, sr=fp.SR, mono=True, offset=offset, duration=clip_seconds)
            sf.write(out_path, y, fp.SR)
        if os.path.exists(out_path):
            samples.append(out_path)
    return samples


@st.cache_data(show_spinner=False)
def song_thumbnail(song_id: str, peaks_key, n_frames, n_freq_bins):
    """Small constellation thumbnail for the library grid card."""
    color = PALETTE[hash(song_id) % len(PALETTE)]
    peaks = peaks_key
    fig, ax = plt.subplots(figsize=(2.6, 1.5), dpi=110)
    fig.patch.set_alpha(0)
    ax.set_facecolor("#0d1414")
    if peaks:
        ts = [p[1] for p in peaks]
        fs = [p[0] for p in peaks]
        ax.scatter(ts, fs, s=2.2, c=color, alpha=0.85, linewidths=0)
    ax.set_xlim(0, max(n_frames, 1))
    ax.set_ylim(0, max(n_freq_bins, 1))
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#0d1414")
    plt.close(fig)
    buf.seek(0)
    return buf


def style_dark_axes(ax):
    ax.set_facecolor("#0d1414")
    ax.tick_params(colors="#6f7a7a", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1d2828")
    ax.xaxis.label.set_color("#9aa6a6")
    ax.yaxis.label.set_color("#9aa6a6")
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)


def plot_spectrogram(mag_db, freqs, times):
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=120)
    fig.patch.set_alpha(0)
    ax.imshow(mag_db, origin="lower", aspect="auto", cmap="magma",
              extent=[times[0] if len(times) else 0, times[-1] if len(times) else 1,
                      freqs[0] if len(freqs) else 0, freqs[-1] if len(freqs) else 1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    style_dark_axes(ax)
    fig.tight_layout()
    return fig


def plot_constellation(peaks, freqs, times, n_peaks_label=True):
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=120)
    fig.patch.set_alpha(0)
    if peaks:
        ts = [times[p[1]] if p[1] < len(times) else p[1] for p in peaks]
        fs = [freqs[p[0]] if p[0] < len(freqs) else p[0] for p in peaks]
        ax.scatter(ts, fs, s=6, c=TEAL, alpha=0.9, linewidths=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    style_dark_axes(ax)
    if n_peaks_label:
        ax.text(0.98, 0.95, f"{len(peaks)} peaks", transform=ax.transAxes,
                ha="right", va="top", color=TEAL, fontsize=9)
    fig.tight_layout()
    return fig


def plot_full_song_fingerprint(peaks, n_frames, highlight_start, highlight_len):
    fig, ax = plt.subplots(figsize=(10.6, 3.0), dpi=120)
    fig.patch.set_alpha(0)
    if peaks:
        ts = [p[1] for p in peaks]
        fs = [p[0] for p in peaks]
        ax.scatter(ts, fs, s=3, c=TEAL, alpha=0.55, linewidths=0)
    if highlight_start is not None:
        ax.axvspan(highlight_start, highlight_start + highlight_len, color=ORANGE, alpha=0.18)
        ax.axvline(highlight_start, color=ORANGE, alpha=0.7, lw=1)
        ax.axvline(highlight_start + highlight_len, color=ORANGE, alpha=0.7, lw=1)
    ax.set_xlim(0, max(n_frames, 1))
    ax.set_xlabel("time (frames)")
    ax.set_ylabel("freq bin")
    style_dark_axes(ax)
    fig.tight_layout()
    return fig


def plot_offset_histogram(hist: dict, best_offset: int, best_count: int):
    fig, ax = plt.subplots(figsize=(10.6, 3.0), dpi=120)
    fig.patch.set_alpha(0)
    if hist:
        offsets = np.array(sorted(hist.keys()))
        counts = np.array([hist[o] for o in offsets])
        colors = [ORANGE if o == best_offset else "#2a5a52" for o in offsets]
        ax.bar(offsets, counts, color=colors, width=max(1, (offsets.max() - offsets.min()) / 400 if len(offsets) > 1 else 1))
        if best_count > 0:
            ax.annotate(f"{best_count:,} hashes\nalign here",
                        xy=(best_offset, best_count), xytext=(best_offset + (offsets.max()-offsets.min())*0.08 + 5, best_count*0.85),
                        color=ORANGE, fontsize=8.5)
    ax.set_xlabel("time offset (database frame − query frame)")
    ax.set_ylabel("# hashes")
    style_dark_axes(ax)
    fig.tight_layout()
    return fig


def run_pipeline(y, db: "fp.FingerprintDB"):
    """Runs stages 1-3 (already timed inside fingerprint_audio), then DB lookup + scoring."""
    fpr = fp.fingerprint_audio(y)

    t0 = time.perf_counter()
    ranked, hist = db.match(fpr.hashes, top_n=5)
    timings_lookup = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    best = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    ratio = (best["score"] / runner_up["score"]) if (best and runner_up and runner_up["score"] > 0) else None
    timings_score = (time.perf_counter() - t0) * 1000

    timings = dict(fpr.timings)
    timings["db_lookup"] = timings_lookup
    timings["scoring"] = timings_score
    return fpr, ranked, hist, best, runner_up, ratio, timings


def render_pipeline_stats(timings, fpr, db, ranked):
    total = sum(timings.values())
    n_hashes_q = len(fpr.hashes)
    n_tracks = len(db.song_meta)
    n_dbhashes = sum(m["n_hashes"] for m in db.song_meta.values())
    cols = st.columns(6)
    items = [
        ("① SPECTROGRAM", f'{timings.get("spectrogram",0):.0f} ms', f'{fpr.mag_db.shape[0]}×{fpr.mag_db.shape[1]}'),
        ("② CONSTELLATION", f'{timings.get("constellation",0):.0f} ms', f'{len(fpr.peaks)} peaks'),
        ("③ HASHING", f'{timings.get("hashing",0):.0f} ms', f'{n_hashes_q:,} hashes'),
        ("④ DB LOOKUP", f'{timings.get("db_lookup",0):.0f} ms', f'{n_tracks} tracks'),
        ("⑤ SCORING", f'{timings.get("scoring",0):.0f} ms', f'offset {ranked[0]["offset"] if ranked else "—"}'),
    ]
    for c, (lbl, num, sub) in zip(cols[:5], items):
        c.markdown(f"""<div class="stat-box"><div class="lbl">{lbl}</div>
                    <div class="num">{num}</div><div class="sub">{sub}</div></div>""", unsafe_allow_html=True)
    cols[5].markdown(f"""<div style="padding-top:18px; text-align:right; color:#6f7a7a; font-size:0.85rem;">
                      total <b style="color:#eef2f2;">{total:.0f} ms</b></div>""", unsafe_allow_html=True)


def render_match_result(fpr, ranked, hist, best, runner_up, ratio, db, query_dur_frames):
    if best is None or best["score"] < MIN_MATCH_SCORE:
        st.markdown(f"""<div class="match-panel">
            <div class="lbl">NO CONFIDENT MATCH</div>
            <div class="name" style="font-size:1.4rem;">Nothing in the library cleared the confidence threshold</div>
            <div class="sub">Best candidate scored only {best["score"] if best else 0} aligned hashes — try a longer or louder clip.</div>
        </div>""", unsafe_allow_html=True)
    else:
        ratio_txt = f'{ratio:.0f}×' if ratio else "—"
        st.markdown(f"""<div class="match-panel">
            <div class="lbl">MATCH FOUND</div>
            <div class="name">{best['name']}</div>
            <div class="sub">cluster score <b>{best['score']:,}</b> · {ratio_txt} the runner-up</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">CANDIDATE SCORES</div>', unsafe_allow_html=True)
    if ranked:
        max_score = max(r["score"] for r in ranked)
        for r in ranked:
            pct = int(100 * r["score"] / max_score) if max_score else 0
            c1, c2, c3 = st.columns([3, 6, 1])
            c1.markdown(f"<div style='padding-top:6px;color:#d7dede;font-size:0.85rem'>{r['name']}</div>", unsafe_allow_html=True)
            c2.markdown(f"""<div style="background:#16201f; border-radius:5px; height:18px; margin-top:8px;">
                <div style="background:{TEAL if r is best else '#2a5a52'}; width:{pct}%; height:18px; border-radius:5px;"></div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"<div style='padding-top:6px;color:#9aa6a6;font-size:0.85rem;text-align:right'>{r['score']:,}</div>", unsafe_allow_html=True)
    else:
        st.caption("No candidate cleared even a single matching hash.")

    # ---- Step 1 ----
    st.markdown("""<div class="step-bar"><div class="step-eyebrow">STEP 1 · FEATURE EXTRACTION</div>
        <div class="step-title">From spectrogram to constellation</div>
        <div class="step-desc">The clip was converted into a time-frequency map (left); brighter means louder at that
        frequency and moment. From that rich image, only the <b>{} most prominent peaks</b> were kept (right).
        Discarding amplitude and phase makes the fingerprint robust to EQ, volume changes, and mild noise.</div>
        </div>""".format(len(fpr.peaks)), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_spectrogram(fpr.mag_db, fpr.freqs, fpr.times), use_container_width=True)
    with c2:
        st.pyplot(plot_constellation(fpr.peaks, fpr.freqs, fpr.times), use_container_width=True)

    # ---- Step 2 ----
    if best is not None:
        meta = db.song_meta[best["song_id"]]
        st.markdown("""<div class="step-bar"><div class="step-eyebrow">STEP 2 · DATABASE SEARCH</div>
            <div class="step-title">Where in the song?</div>
            <div class="step-desc">The <b>{:,} fingerprint hashes</b> were looked up against every indexed track.
            Below is the full fingerprint of <b>{}</b> reconstructed from the database, each dot is a stored hash
            anchor. The highlighted window is exactly where the query clip sits inside the full song.</div>
            </div>""".format(len(fpr.hashes), best["name"]), unsafe_allow_html=True)
        st.pyplot(plot_full_song_fingerprint(meta["peaks"], meta["n_frames"],
                                              best["offset"], query_dur_frames), use_container_width=True)

        # ---- Step 3 ----
        st.markdown("""<div class="step-bar"><div class="step-eyebrow">STEP 3 · THE PROOF</div>
            <div class="step-title">The alignment spike</div>
            <div class="step-desc">Every matched hash votes for a time offset (database frame minus query frame).
            Chance matches scatter votes randomly, forming a flat noise floor. A genuine match makes them converge:
            <b>{:,} hashes agreed on a single offset</b>. That spike cannot be a coincidence.</div>
            </div>""".format(best["score"]), unsafe_allow_html=True)
        st.pyplot(plot_offset_histogram(hist.get(best["song_id"], {}), best["offset"], best["score"]),
                  use_container_width=True)


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_library, tab_identify, tab_batch = st.tabs(["◆ LIBRARY", "◎ IDENTIFY", "▦ BATCH"])

# ===================== LIBRARY =====================
with tab_library:
    db = load_database()
    st.markdown('<div class="eyebrow">LIBRARY</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card">Song indexing is managed by the admin.<br>
        Upload a clip in the Identify tab (or a batch in the Batch tab) to test the library.</div>""",
        unsafe_allow_html=True)
    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">IN THE DATABASE &nbsp;·&nbsp; {len(db.song_meta)} TRACKS</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

    song_ids = list(db.song_meta.keys())
    n_cols = 4
    for row_start in range(0, len(song_ids), n_cols):
        cols = st.columns(n_cols)
        for col, song_id in zip(cols, song_ids[row_start:row_start + n_cols]):
            meta = db.song_meta[song_id]
            thumb = song_thumbnail(song_id, meta["peaks"], meta["n_frames"], 1 + max((p[0] for p in meta["peaks"]), default=1))
            with col:
                st.image(thumb, use_container_width=True)
                st.markdown(f'<div class="song-name">{meta["name"]}</div>'
                            f'<div class="song-hashes">{meta["n_hashes"]:,} hashes</div>',
                            unsafe_allow_html=True)
                st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

# ===================== IDENTIFY =====================

with tab_identify:
    db = load_database()
    st.markdown('<div class="eyebrow">SEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title" style="font-size:1.5rem;">Identify a clip</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload", type=["wav", "mp3", "flac", "ogg", "m4a"],
                                 label_visibility="collapsed",
                                 help="200MB per file • WAV, MP3, FLAC, OGG, M4A")

    # 1. Initialize State Variables (Callbacks need these to exist)
    if "active_sample" not in st.session_state:
        st.session_state["active_sample"] = None
    if "run_immediate" not in st.session_state:
        st.session_state["run_immediate"] = False

    # 2. Callback Function: Updates state BEFORE the UI redraws
    def trigger_sample(path):
        st.session_state["active_sample"] = path
        st.session_state["run_immediate"] = True

    chosen_bytes = None

    # Handle Sample Tracks (auto-generated from the indexed library, no manual setup needed)
    sig = fp.folder_fingerprint(SONG_DB_FOLDER)
    sample_files = get_or_build_samples(sig)
    if sample_files:
        st.markdown('<div class="eyebrow" style="margin-top:10px;">OR TRY A SAMPLE</div>', unsafe_allow_html=True)
        for spath in sample_files:
            sname = os.path.splitext(os.path.basename(spath))[0]

            # Check active state
            is_active = (st.session_state["active_sample"] == spath)

            c1, c2 = st.columns([5, 1])
            with c1:
                if is_active:
                    # Draw the highlight!
                    st.markdown(f"<div style='color: {TEAL}; font-size: 0.85rem; font-weight: 700; margin-bottom: -10px;'>▶ TARGET TRACK: {sname}</div>", unsafe_allow_html=True)
                st.audio(spath)
            with c2:
                if is_active:
                    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True) 
                else:
                    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                
                # Attach the callback here
                st.button("Try", key=f"try_{sname}", type="primary" if is_active else "secondary",
                          on_click=trigger_sample, args=(spath,))

    # Handle File Uploads (Overrides active sample)
    if uploaded is not None:
        chosen_bytes = uploaded.read()
        st.session_state["active_sample"] = None 

    st.markdown("<br>", unsafe_allow_html=True)
    identify_clicked = st.button("Identify", type="primary")

    # EXECUTE IF: Main button is clicked OR a sample "Try" button triggered the flag
    if identify_clicked or st.session_state["run_immediate"]:
        # Reset the flag so it doesn't auto-run if the user clicks other tabs later
        st.session_state["run_immediate"] = False 
        
        target_path = st.session_state["active_sample"]
        
        if chosen_bytes is None and target_path is None:
            st.warning("Upload a clip or pick a sample first.")
        else:
            with st.spinner("Fingerprinting clip…"):
                if chosen_bytes is not None:
                    tmp_path = os.path.join(APP_DIR, "_tmp_query")
                    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
                    tmp_path += suffix
                    with open(tmp_path, "wb") as fh:
                        fh.write(chosen_bytes)
                    y = fp.load_audio(tmp_path)
                    os.remove(tmp_path)
                else:
                    y = fp.load_audio(target_path)

                fpr, ranked, hist, best, runner_up, ratio, timings = run_pipeline(y, db)

            render_pipeline_stats(timings, fpr, db, ranked)
            st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
            query_dur_frames = fpr.mag_db.shape[1]
            render_match_result(fpr, ranked, hist, best, runner_up, ratio, db, query_dur_frames)

# ===================== BATCH =====================
with tab_batch:
    db = load_database()
    st.markdown('<div class="eyebrow">BATCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title" style="font-size:1.5rem;">Identify many clips at once</div>', unsafe_allow_html=True)
    st.markdown("""<div class="step-desc" style="margin-bottom:14px;">
        Upload a set of query clips. Each is identified against the <b>currently indexed library</b>,
        and the results are written to a standardised <code>results.csv</code> with columns
        <code>filename, prediction</code>. The <code>prediction</code> is the matched track's filename
        without its extension, or <code>none</code> when no candidate clears the confidence threshold.
        </div>""", unsafe_allow_html=True)

    batch_files = st.file_uploader("Upload many", type=["wav", "mp3", "flac", "ogg", "m4a"],
                                    accept_multiple_files=True, label_visibility="collapsed")

    run_batch = st.button("Run batch", type="primary")

    if run_batch:
        if not batch_files:
            st.warning("Upload at least one clip first.")
        else:
            rows = []
            progress = st.progress(0.0, text="Running batch…")
            for i, f in enumerate(batch_files):
                tmp_path = os.path.join(APP_DIR, f"_tmp_batch_{i}")
                suffix = os.path.splitext(f.name)[1] or ".wav"
                tmp_path += suffix
                with open(tmp_path, "wb") as fh:
                    fh.write(f.read())
                try:
                    y = fp.load_audio(tmp_path)
                    fpr = fp.fingerprint_audio(y)
                    ranked, _ = db.match(fpr.hashes, top_n=1)
                    if ranked and ranked[0]["score"] >= MIN_MATCH_SCORE:
                        pred = ranked[0]["song_id"]
                    else:
                        pred = "none"
                except Exception:
                    pred = "none"
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                rows.append({"filename": f.name, "prediction": pred})
                progress.progress((i + 1) / len(batch_files), text=f"Running batch… {f.name}")
            progress.empty()

            df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.markdown('<div class="eyebrow" style="margin-top:6px;">RESULTS</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

            n_matched = (df["prediction"] != "none").sum()
            st.caption(f"{n_matched} / {len(df)} clips matched to a track ({len(df)-n_matched} returned `none`).")

            csv_bytes = df.to_csv(index=False).encode()
            st.download_button("⬇ Download results.csv", data=csv_bytes,
                                file_name="results.csv", mime="text/csv")
