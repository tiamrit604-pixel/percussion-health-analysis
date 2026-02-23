from pydub import AudioSegment
import streamlit as st
import numpy as np
import librosa
import librosa.display

from spafe.features import mfcc
from spafe.utils.preprocessing import SlidingWindow
import spafe.utils.vis as vis
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, stft, periodogram
from scipy.fft import fft, fftfreq
import pywt
import io

st.set_page_config(layout="wide", page_title="Percussion Health Analysis - Advanced")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-title {
        color: #e94560;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .main-subtitle {
        color: #a8b2d8;
        font-size: 1.1rem;
        font-weight: 400;
    }
    .hit-card {
        border: 2px solid #333;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .hit-card-selected {
        border: 2px solid #00ff88;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: rgba(0,255,136,0.05);
    }
    .compare-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><div class="main-title">🎯 Percussion Health Analysis - Advanced</div><div class="main-subtitle">Multi-Hit Recording & Selection System</div></div>', unsafe_allow_html=True)

# ----------------------------- 
# Session State Initialization
# -----------------------------
if "multi_recordings" not in st.session_state:
    st.session_state.multi_recordings = {"healthy": [], "unhealthy": []}
if "detected_hits" not in st.session_state:
    st.session_state.detected_hits = None
if "selected_hit_index" not in st.session_state:
    st.session_state.selected_hit_index = None
if "current_cell_type" not in st.session_state:
    st.session_state.current_cell_type = "Healthy"

# ----------------------------- 
# Sidebar Configuration
# -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    cell_type = st.radio(
        "Select Cell Type:",
        ["Healthy", "Unhealthy"],
        help="Choose whether you're testing a healthy or unhealthy cell"
    )
    st.session_state.current_cell_type = cell_type

    st.markdown("---")
    st.markdown("## 📊 Recording Stats")
    st.metric("Healthy Recordings", len(st.session_state.multi_recordings["healthy"]))
    st.metric("Unhealthy Recordings", len(st.session_state.multi_recordings["unhealthy"]))

    st.markdown("---")
    st.markdown("## 🎛️ Detection Settings")
    peak_threshold = st.slider(
        "Peak Detection Threshold",
        min_value=0.01, max_value=0.5, value=0.05, step=0.01,
        help="Minimum amplitude required to detect a hit"
    )
    min_hit_distance = st.slider(
        "Minimum Hit Distance (seconds)",
        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        help="Minimum time between consecutive hits"
    )
    hit_duration = st.slider(
        "Hit Duration (seconds)",
        min_value=0.2, max_value=1.0, value=0.5, step=0.1,
        help="Duration to extract after each hit"
    )

    st.markdown("---")
    if st.button("🗑️ Clear All Data"):
        st.session_state.multi_recordings = {"healthy": [], "unhealthy": []}
        st.session_state.detected_hits = None
        st.session_state.selected_hit_index = None
        st.success("All data cleared!")
        st.rerun()

# ----------------------------- 
# Instructions
# -----------------------------
with st.expander("📋 How to Use This Tool", expanded=True):
    st.markdown("""
    ### **New Workflow with Multi-Hit Detection:**

    #### **Step 1: Record Multiple Hits**
    1. Select cell type (Healthy or Unhealthy)
    2. Click the microphone button
    3. **Hit the plate 5 times** with pauses between each hit
    4. Stop recording after all hits

    #### **Step 2: Automatic Hit Detection**
    - The system will automatically detect all 5 hits
    - Each hit will be extracted and displayed individually
    - You'll see time-series plots for each detected hit

    #### **Step 3: Select the Best Hit**
    - Review all detected hits in the time-series view
    - Click "Select Hit #X" button for the clearest hit
    - Avoid hits with noise or weak impacts

    #### **Step 4: Analyze Selected Hit**
    - Click "Analyze Selected Hit"
    - View comprehensive signal processing results

    #### **Step 5: Compare Recordings**
    - Use the comparison section to compare any two recordings
    - Compare healthy vs unhealthy, or healthy vs healthy, etc.
    - View side-by-side plots for all analysis types

    ### **Tips for Best Results:**
    - 🎯 Hit the center of each cell consistently
    - ⏱️ Wait ~1 second between hits
    - 💪 Use consistent force for all hits
    - 🔇 Record in a quiet environment
    - 🎤 Keep microphone 10-15 cm from plate
    """)

# =========================================================
# UNIVERSAL AUDIO LOADER (accept mp3, wav, m4a, mp4 etc)
# =========================================================
def load_audio_any_format(file_bytes):
    """Load ANY audio/video format (wav, mp3, m4a, mp4 etc)."""
    try:
        # First try librosa (fast)
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
        return y, sr
    except:
        # If librosa fails → use pydub + ffmpeg
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        sr = audio.frame_rate

        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        # stereo → mono
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.mean(axis=1)

        # normalize
        samples = samples / np.max(np.abs(samples))

        return samples, sr
# ----------------------------- 
# Hit Detection Function
# ----------------------------- 
def detect_multiple_hits(y, sr, height_threshold=0.05, min_distance=0.5, duration=0.5):
    """
    Detect multiple hits using find_peaks on abs(y), matching notebook approach:
    height = 0.3 * max(abs(y)) or user threshold, distance = min_distance * sr,
    window pre-peak 0.02s, post-peak = duration, normalize each segment.
    """
    abs_y = np.abs(y)
    effective_height = max(height_threshold, 0.3 * np.max(abs_y))
    min_distance_samples = int(min_distance * sr)

    peaks, _ = find_peaks(
        abs_y,
        height=effective_height,
        distance=min_distance_samples
    )

    hits = []
    pre_samples = int(0.02 * sr)
    duration_samples = int(duration * sr)

    for i, peak in enumerate(peaks):
        start = max(0, peak - pre_samples)
        end = peak + duration_samples
        hit_signal = y[start:min(end, len(y))]

        if len(hit_signal) == 0:
            continue

        # Normalize each segment by its max (as in notebook)
        max_val = np.max(np.abs(hit_signal))
        if max_val > 0:
            hit_signal = hit_signal / max_val

        rms = np.sqrt(np.mean(hit_signal**2))
        peak_amplitude = np.max(np.abs(hit_signal))

        hits.append({
            'index': i,
            'onset_sample': peak,
            'onset_time': peak / sr,
            'signal': hit_signal,
            'rms': rms,
            'peak_amplitude': peak_amplitude,
            'duration': len(hit_signal) / sr
        })

    return hits

# =========================================================
# SIGNAL PROCESSING HELPERS  (notebook-exact methods)
# =========================================================

def _plot_time_domain(ax, y, sr, title):
    """Time domain — numpy linspace + simple plot (Cell 5)."""
    time = np.linspace(0, len(y) / sr, len(y))
    ax.plot(time, y, linewidth=0.8, color='#1f77b4')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)


def _plot_fft(ax, y, sr, title):
    """Single-sided amplitude spectrum — scipy.fft (Cell 6)."""
    N = len(y)
    yf = fft(y.astype(np.float64))
    P2 = np.abs(yf / N)
    P1 = P2[:N // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    xf = fftfreq(N, 1 / sr)[:N // 2 + 1]

    peak_freq = xf[np.argmax(P1)]
    ax.plot(xf, P1, linewidth=0.8, color='#ff7f0e')
    ax.axvline(x=peak_freq, color='red', linestyle='--', linewidth=1,
               alpha=0.7, label=f'Peak: {peak_freq:.1f} Hz')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, min(10000, sr / 2))
    ax.grid(True)
    ax.legend(fontsize=9)
    return peak_freq


def _plot_stft(ax, y, sr, title):
    """STFT spectrogram — scipy.signal.stft + pcolormesh (Cell 7)."""
    f, t, Zxx = stft(y.astype(np.float64), sr, nperseg=256, noverlap=200, nfft=512)
    pcm = ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylim(0, min(10000, sr / 2))
    return pcm


def _plot_psd(ax, y, sr, title):
    """PSD — scipy.signal.periodogram with normalized frequency (Cell 8)."""
    x = y.astype(np.float64)
    f, pxx = periodogram(
        x,
        fs=1.0,
        window='boxcar',
        nfft=4096,
        detrend=False,
        scaling='density',
        return_onesided=True
    )
    ax.plot(f, pxx, linewidth=1.2, color='#2ca02c')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Power/Frequency")
    ax.set_xlim(0, 0.5)
    ax.grid(True)


def _plot_cwt(fig, ax, y, sr, title):
    """CWT scalogram — pywt 'morl', scales 1-256, imshow (Cell 9)."""
    x = y.astype(np.float64)
    scales = np.arange(1, 256)
    coefficients, frequencies = pywt.cwt(x, scales, 'morl', sampling_period=1 / sr)
    im = ax.imshow(
        np.abs(coefficients),
        extent=[0, len(x) / sr, frequencies[-1], frequencies[0]],
        aspect='auto',
        cmap='jet'
    )
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label='Magnitude')


def _plot_mfcc(fig, ax, y, sr, title):
    """MFCC using SPAFE instead of librosa."""
    x = y.astype(np.float64)

    # SPAFE MFCC extraction
    mfccs = mfcc.imfcc(
        x,
        fs=sr,
        num_ceps=13,
        window=SlidingWindow(win_len=0.025, win_hop=0.01, win_type='hamming'),
        nfilts=128,
        nfft=2048,
        low_freq=200,
        high_freq=15000,
        scale='constant',
        dct_type=2,
        lifter=22
    )

    # Plot
    im = ax.imshow(
        mfccs.T,
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )

    fig.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel("MFCC Coefficient")


# =========================================================
# SINGLE RECORDING ANALYSIS
# =========================================================

def analyze_audio(y, sr, cell_type_label, metadata=None):
    """Comprehensive audio signal analysis using notebook-exact methods."""
    st.markdown(f"### 🎯 Analysis Results — {cell_type_label} Cell")

    if metadata:
        st.info(
            f"📍 Analyzing Hit #{metadata.get('hit_number', 'N/A')} | "
            f"Peak: {metadata.get('peak_amplitude', 0):.3f} | "
            f"RMS: {metadata.get('rms', 0):.4f}"
        )

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Duration", f"{len(y)/sr:.2f} sec")
    with info_col2:
        st.metric("Sample Rate", f"{sr} Hz")
    with info_col3:
        st.metric("Samples", f"{len(y):,}")

    st.markdown("---")

    # ── 1. TIME DOMAIN ──────────────────────────────────────────────
    st.markdown("#### 📈 Time Domain")
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    _plot_time_domain(ax1, y, sr, f"Time Domain Signal — {cell_type_label} Cell")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    # ── 2. FFT ──────────────────────────────────────────────────────
    st.markdown("#### 🎵 Single-Sided Amplitude Spectrum (FFT)")
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    peak_freq = _plot_fft(ax2, y, sr, f"FFT Spectrum — {cell_type_label} Cell")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.info(f"🎵 **Dominant Frequency**: {peak_freq:.1f} Hz")

    # ── 3. STFT ─────────────────────────────────────────────────────
    st.markdown("#### 🌈 STFT Spectrogram")
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    pcm3 = _plot_stft(ax3, y, sr, f"STFT Spectrogram — {cell_type_label} Cell")
    fig3.colorbar(pcm3, ax=ax3, label="Magnitude")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # ── 4. PSD ──────────────────────────────────────────────────────
    st.markdown("#### ⚡ Power Spectral Density")
    fig4, ax4 = plt.subplots(figsize=(11, 4))
    _plot_psd(ax4, y, sr, f"Power Spectral Density — {cell_type_label} Cell")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # ── 5. CWT ──────────────────────────────────────────────────────
    st.markdown("#### 🌊 CWT Scalogram")
    fig5, ax5 = plt.subplots(figsize=(11, 5))
    _plot_cwt(fig5, ax5, y, sr, f"CWT Scalogram — {cell_type_label} Cell")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    # ── 6. MFCC ─────────────────────────────────────────────────────
    st.markdown("#### 🎤 MFCC")
    fig6, ax6 = plt.subplots(figsize=(11, 5))
    _plot_mfcc(fig6, ax6, y, sr, f"MFCC — {cell_type_label} Cell")
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    # ── 7. FEATURE SUMMARY ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Feature Summary")
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms_feat = librosa.feature.rms(y=y)[0]
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.metric("Zero Crossing Rate", f"{np.mean(zcr):.4f}")
    with mc2:
        st.metric("RMS Energy", f"{np.mean(rms_feat):.4f}")
    with mc3:
        st.metric("Spectral Centroid", f"{np.mean(spec_cent):.1f} Hz")
    with mc4:
        st.metric("Spectral Rolloff", f"{np.mean(spec_rolloff):.1f} Hz")
    with mc5:
        st.metric("Spectral Bandwidth", f"{np.mean(spec_bw):.1f} Hz")


# =========================================================
# COMPARISON FUNCTION
# =========================================================

def compare_two_recordings(rec1_data, rec1_label, rec2_data, rec2_label):
    """Compare two recordings side-by-side using notebook-exact methods."""
    st.markdown(f"### 📊 Comparison: {rec1_label} vs {rec2_label}")

    # Extract signals
    if 'signal' in rec1_data:
        y1, sr1 = rec1_data['signal'], rec1_data['sr']
        meta1 = rec1_data.get('metadata', {})
    else:
        audio_data = rec1_data.get('data', rec1_data)
        if hasattr(audio_data, 'read'):
            audio_data = audio_data.read()
        y1, sr1 = librosa.load(io.BytesIO(audio_data), sr=None)
        meta1 = {}

    if 'signal' in rec2_data:
        y2, sr2 = rec2_data['signal'], rec2_data['sr']
        meta2 = rec2_data.get('metadata', {})
    else:
        audio_data = rec2_data.get('data', rec2_data)
        if hasattr(audio_data, 'read'):
            audio_data = audio_data.read()
        y2, sr2 = librosa.load(io.BytesIO(audio_data), sr=None)
        meta2 = {}

    # ── Metadata summary ────────────────────────────────────────────
    st.markdown("#### 📋 Recording Metadata")
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        st.markdown(f"**{rec1_label}**")
        st.metric("Duration", f"{len(y1)/sr1:.2f} sec")
        st.metric("Sample Rate", f"{sr1} Hz")
        st.metric("Peak Amplitude", f"{meta1.get('peak_amplitude', np.max(np.abs(y1))):.3f}")
        st.metric("RMS", f"{meta1.get('rms', np.sqrt(np.mean(y1**2))):.4f}")
    with meta_col2:
        st.markdown(f"**{rec2_label}**")
        st.metric("Duration", f"{len(y2)/sr2:.2f} sec")
        st.metric("Sample Rate", f"{sr2} Hz")
        st.metric("Peak Amplitude", f"{meta2.get('peak_amplitude', np.max(np.abs(y2))):.3f}")
        st.metric("RMS", f"{meta2.get('rms', np.sqrt(np.mean(y2**2))):.4f}")

    st.markdown("---")

    # ── 1. TIME DOMAIN ──────────────────────────────────────────────
    st.markdown("#### 📈 Time Domain Waveforms")
    fig1, (ax1L, ax1R) = plt.subplots(1, 2, figsize=(18, 4))
    _plot_time_domain(ax1L, y1, sr1, rec1_label)
    _plot_time_domain(ax1R, y2, sr2, rec2_label)
    ax1R.lines[0].set_color('#ff7f0e')
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    # ── 2. FFT ──────────────────────────────────────────────────────
    st.markdown("#### 🎵 Single-Sided Amplitude Spectrum (FFT)")
    fig2, (ax2L, ax2R) = plt.subplots(1, 2, figsize=(18, 4))
    pf1 = _plot_fft(ax2L, y1, sr1, rec1_label)
    pf2 = _plot_fft(ax2R, y2, sr2, rec2_label)
    # Recolor right panel orange
    ax2R.lines[0].set_color('#ff7f0e')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.info(f"🎵 **{rec1_label}** dominant: {pf1:.1f} Hz  |  **{rec2_label}** dominant: {pf2:.1f} Hz")

    # ── 3. STFT ─────────────────────────────────────────────────────
    st.markdown("#### 🌈 STFT Spectrograms")
    fig3, (ax3L, ax3R) = plt.subplots(1, 2, figsize=(18, 5))
    pcm3L = _plot_stft(ax3L, y1, sr1, rec1_label)
    pcm3R = _plot_stft(ax3R, y2, sr2, rec2_label)
    fig3.colorbar(pcm3L, ax=ax3L, label="Magnitude")
    fig3.colorbar(pcm3R, ax=ax3R, label="Magnitude")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # ── 4. PSD ──────────────────────────────────────────────────────
    st.markdown("#### ⚡ Power Spectral Density")
    fig4, (ax4L, ax4R) = plt.subplots(1, 2, figsize=(18, 4))
    _plot_psd(ax4L, y1, sr1, rec1_label)
    _plot_psd(ax4R, y2, sr2, rec2_label)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # ── 5. CWT ──────────────────────────────────────────────────────
    st.markdown("#### 🌊 CWT Scalograms")
    fig5, (ax5L, ax5R) = plt.subplots(1, 2, figsize=(18, 5))
    _plot_cwt(fig5, ax5L, y1, sr1, rec1_label)
    _plot_cwt(fig5, ax5R, y2, sr2, rec2_label)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    # ── 6. MFCC ─────────────────────────────────────────────────────
    st.markdown("#### 🎤 MFCC Comparison")
    fig6, (ax6L, ax6R) = plt.subplots(1, 2, figsize=(18, 5))
    _plot_mfcc(fig6, ax6L, y1, sr1, rec1_label)
    _plot_mfcc(fig6, ax6R, y2, sr2, rec2_label)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    # ── 7. FEATURE SUMMARY ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Feature Summary Comparison")

    zcr1 = librosa.feature.zero_crossing_rate(y1)[0]
    rms1 = librosa.feature.rms(y=y1)[0]
    cent1 = librosa.feature.spectral_centroid(y=y1, sr=sr1)[0]
    roll1 = librosa.feature.spectral_rolloff(y=y1, sr=sr1)[0]
    bw1 = librosa.feature.spectral_bandwidth(y=y1, sr=sr1)[0]

    zcr2 = librosa.feature.zero_crossing_rate(y2)[0]
    rms2 = librosa.feature.rms(y=y2)[0]
    cent2 = librosa.feature.spectral_centroid(y=y2, sr=sr2)[0]
    roll2 = librosa.feature.spectral_rolloff(y=y2, sr=sr2)[0]
    bw2 = librosa.feature.spectral_bandwidth(y=y2, sr=sr2)[0]

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f"**{rec1_label}**")
        st.metric("Zero Crossing Rate", f"{np.mean(zcr1):.4f}")
        st.metric("RMS Energy", f"{np.mean(rms1):.4f}")
        st.metric("Spectral Centroid", f"{np.mean(cent1):.1f} Hz")
        st.metric("Spectral Rolloff", f"{np.mean(roll1):.1f} Hz")
        st.metric("Spectral Bandwidth", f"{np.mean(bw1):.1f} Hz")
    with fc2:
        st.markdown(f"**{rec2_label}**")
        st.metric("Zero Crossing Rate", f"{np.mean(zcr2):.4f}", delta=f"{np.mean(zcr2)-np.mean(zcr1):.4f}")
        st.metric("RMS Energy", f"{np.mean(rms2):.4f}", delta=f"{np.mean(rms2)-np.mean(rms1):.4f}")
        st.metric("Spectral Centroid", f"{np.mean(cent2):.1f} Hz", delta=f"{np.mean(cent2)-np.mean(cent1):.1f} Hz")
        st.metric("Spectral Rolloff", f"{np.mean(roll2):.1f} Hz", delta=f"{np.mean(roll2)-np.mean(roll1):.1f} Hz")
        st.metric("Spectral Bandwidth", f"{np.mean(bw2):.1f} Hz", delta=f"{np.mean(bw2)-np.mean(bw1):.1f} Hz")


# =========================================================
# RECORDING SECTION
# =========================================================

st.markdown(f"## 📂 Step 1: Upload Audio File — {cell_type} Cell")

col_rec1, col_rec2 = st.columns([2, 1])

with col_rec1:
    st.info(f"📍 Currently uploading for: **{cell_type}** cell")
    st.markdown("Upload a `.wav`` file containing **several hits**.")
    uploaded_file = st.file_uploader(
        f"Upload audio file for {cell_type} cell",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        key=f"uploader_{cell_type}"
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file
        st.success(f"✅ File **{uploaded_file.name}** loaded! Click 'Process' below.")
        st.audio(uploaded_file)
    else:
        audio_bytes = None

if audio_bytes and st.button("🔍 Process File & Detect Hits", type="primary", key="process_btn"):
    try:
        audio_bytes.seek(0)
        audio_data = audio_bytes.read()
        audio_bytes.seek(0)
        y, sr = load_audio_any_format(audio_data)
        st.info(f"🎛️ **Sampling Rate: {sr} Hz**")

        if len(y) == 0:
            st.error("❌ Recording is empty!")
        else:
            with st.spinner("🔍 Detecting hits..."):
                hits = detect_multiple_hits(
                    y, sr,
                    height_threshold=peak_threshold,
                    min_distance=min_hit_distance,
                    duration=hit_duration
                )

            if len(hits) == 0:
                st.error("❌ No hits detected! Try hitting harder or adjusting detection settings.")
            else:
                st.session_state.detected_hits = {
                    'hits': hits,
                    'full_signal': y,
                    'sr': sr,
                    'cell_type': cell_type
                }
                st.session_state.selected_hit_index = None
                st.success(f"✅ Detected {len(hits)} hit(s)!")
                st.balloons()
    except Exception as e:
        st.error(f"❌ Error processing recording: {str(e)}")
        with st.expander("🐛 See detailed error"):
            st.exception(e)

with col_rec2:
    st.markdown("### 📝 File Info")
    if uploaded_file is not None:
        st.info(f"📄 **{uploaded_file.name}**")
        st.info(f"📦 Size: {uploaded_file.size / 1024:.1f} KB")
        st.info(f"🎵 Type: {uploaded_file.type}")
    else:
        st.warning("No file uploaded yet")

# =========================================================
# DISPLAY DETECTED HITS
# =========================================================

if st.session_state.detected_hits is not None:
    st.markdown("---")
    st.markdown("## 👁️ Step 2: Review Detected Hits & Select Best One")

    hits_data = st.session_state.detected_hits
    hits = hits_data['hits']
    full_signal = hits_data['full_signal']
    sr = hits_data['sr']
    cell_type_recorded = hits_data['cell_type']

    st.info(f"📊 Found **{len(hits)} hit(s)** in the recording. Review each hit and select the best one.")

    # Full recording with markers
    st.markdown("### 🎵 Full Recording with Detected Hits")
    fig_full, ax_full = plt.subplots(figsize=(14, 4))
    time_full = np.linspace(0, len(full_signal) / sr, len(full_signal))
    ax_full.plot(time_full, full_signal, linewidth=0.5, alpha=0.7, color='gray', label='Full Signal')
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    for i, hit in enumerate(hits):
        onset_time = hit['onset_time']
        color = colors[i % len(colors)]
        ax_full.axvline(x=onset_time, color=color, linestyle='--', linewidth=2,
                        label=f'Hit {i+1}', alpha=0.8)
        ax_full.axvspan(onset_time, onset_time + hit['duration'], alpha=0.2, color=color)
    ax_full.set_title(f"Full Recording — {cell_type_recorded} Cell ({len(hits)} hits detected)",
                      fontsize=14, fontweight='bold')
    ax_full.set_xlabel("Time (s)")
    ax_full.set_ylabel("Amplitude")
    ax_full.legend(loc='upper right')
    ax_full.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_full)
    plt.close()

    # Individual hits
    st.markdown("### 🎯 Individual Hit Waveforms")
    n_hits = len(hits)
    n_cols = min(3, n_hits)
    n_rows = (n_hits + n_cols - 1) // n_cols

    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            hit_idx = row * n_cols + col_idx
            if hit_idx < n_hits:
                hit = hits[hit_idx]
                with cols[col_idx]:
                    is_selected = (st.session_state.selected_hit_index == hit_idx)
                    card_class = "hit-card-selected" if is_selected else "hit-card"
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    st.markdown(f"#### Hit #{hit_idx + 1}")
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.metric("Peak", f"{hit['peak_amplitude']:.3f}")
                    with mc2:
                        st.metric("RMS", f"{hit['rms']:.4f}")
                    fig_hit, ax_hit = plt.subplots(figsize=(6, 3))
                    t_hit = np.linspace(0, hit['duration'], len(hit['signal']))
                    ax_hit.plot(t_hit, hit['signal'], linewidth=0.8,
                                color=colors[hit_idx % len(colors)])
                    ax_hit.set_title(f"Hit {hit_idx + 1} @ {hit['onset_time']:.2f}s")
                    ax_hit.set_xlabel("Time (s)")
                    ax_hit.set_ylabel("Amplitude")
                    ax_hit.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_hit)
                    plt.close()
                    button_label = "✅ Selected" if is_selected else f"Select Hit #{hit_idx + 1}"
                    button_type = "secondary" if is_selected else "primary"
                    if st.button(button_label, key=f"select_hit_{hit_idx}",
                                 type=button_type, disabled=is_selected):
                        st.session_state.selected_hit_index = hit_idx
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    # Save selected hit
    st.markdown("---")
    if st.session_state.selected_hit_index is not None:
        selected_hit = hits[st.session_state.selected_hit_index]
        st.success(f"✅ Hit #{st.session_state.selected_hit_index + 1} selected!")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Selected Hit", f"#{st.session_state.selected_hit_index + 1}")
        with sc2:
            st.metric("Peak Amplitude", f"{selected_hit['peak_amplitude']:.3f}")
        with sc3:
            st.metric("RMS Energy", f"{selected_hit['rms']:.4f}")

        if st.button("💾 Save Selected Hit & Continue", type="primary", key="save_hit"):
            cell_key = cell_type_recorded.lower()
            from scipy.io import wavfile
            buffer = io.BytesIO()
            wavfile.write(buffer, sr, (selected_hit['signal'] * 32767).astype(np.int16))
            buffer.seek(0)
            st.session_state.multi_recordings[cell_key].append({
                'data': buffer.read(),
                'sr': sr,
                'signal': selected_hit['signal'],
                'metadata': {
                    'hit_number': st.session_state.selected_hit_index + 1,
                    'peak_amplitude': selected_hit['peak_amplitude'],
                    'rms': selected_hit['rms'],
                    'onset_time': selected_hit['onset_time']
                }
            })
            st.success(f"✅ Hit #{st.session_state.selected_hit_index + 1} saved for {cell_type_recorded} cell!")
            st.balloons()
            st.session_state.detected_hits = None
            st.session_state.selected_hit_index = None
    else:
        st.warning("⚠️ Please select a hit from the options above before continuing.")

# =========================================================
# ANALYSIS SECTION
# =========================================================

st.markdown("---")
st.markdown("## 🔬 Step 3: Analyze Selected Hit")

analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

with analysis_col1:
    analyze_type = st.selectbox("Select cell type to analyze:", ["Healthy", "Unhealthy"])

with analysis_col2:
    cell_key_analyze = analyze_type.lower()
    num_recordings = len(st.session_state.multi_recordings[cell_key_analyze])
    if num_recordings > 0:
        recording_index = st.selectbox(
            "Select recording:",
            range(1, num_recordings + 1),
            format_func=lambda x: f"Recording #{x}"
        )
    else:
        st.warning(f"No {analyze_type} recordings available")
        recording_index = None

with analysis_col3:
    st.write("")
    st.write("")
    analyze_button = st.button(
        "📊 Analyze Selected Recording",
        type="primary",
        disabled=(recording_index is None)
    )

if analyze_button and recording_index is not None:
    selected_recording = st.session_state.multi_recordings[cell_key_analyze][recording_index - 1]
    with st.spinner(f"🔍 Analyzing {analyze_type} cell recording #{recording_index}..."):
        try:
            if 'signal' in selected_recording:
                y = selected_recording['signal']
                sr = selected_recording['sr']
                metadata = selected_recording.get('metadata', None)
            else:
                audio_data = selected_recording.get('data', selected_recording)
                if hasattr(audio_data, 'read'):
                    audio_data = audio_data.read()
                y, sr = librosa.load(io.BytesIO(audio_data), sr=None)
                metadata = None

            if len(y) == 0:
                st.error("❌ Recording is empty.")
            else:
                analyze_audio(y, sr, analyze_type, metadata)
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            with st.expander("🐛 See detailed error"):
                st.exception(e)

# =========================================================
# COMPARISON SECTION
# =========================================================

st.markdown("---")
st.markdown('<div class="compare-section">', unsafe_allow_html=True)
st.markdown("## 🔄 Step 4: Compare Two Recordings")
st.info("📊 **Compare any two recordings side-by-side** — Choose healthy vs unhealthy, healthy vs healthy, or unhealthy vs unhealthy")

comp_col1, comp_col2, comp_col3 = st.columns(3)

with comp_col1:
    st.markdown("### 📍 First Recording")
    comp_type_1 = st.selectbox("Cell Type 1:", ["Healthy", "Unhealthy"], key="comp_type_1")
    cell_key_1 = comp_type_1.lower()
    num_recordings_1 = len(st.session_state.multi_recordings[cell_key_1])
    if num_recordings_1 > 0:
        comp_rec_1 = st.selectbox(
            "Recording 1:", range(1, num_recordings_1 + 1),
            format_func=lambda x: f"Recording #{x}", key="comp_rec_1"
        )
    else:
        st.warning(f"No {comp_type_1} recordings")
        comp_rec_1 = None

with comp_col2:
    st.markdown("### 📍 Second Recording")
    comp_type_2 = st.selectbox("Cell Type 2:", ["Healthy", "Unhealthy"], key="comp_type_2")
    cell_key_2 = comp_type_2.lower()
    num_recordings_2 = len(st.session_state.multi_recordings[cell_key_2])
    if num_recordings_2 > 0:
        comp_rec_2 = st.selectbox(
            "Recording 2:", range(1, num_recordings_2 + 1),
            format_func=lambda x: f"Recording #{x}", key="comp_rec_2"
        )
    else:
        st.warning(f"No {comp_type_2} recordings")
        comp_rec_2 = None

with comp_col3:
    st.markdown("### 🎯 Compare")
    st.write("")
    st.write("")
    compare_button = st.button(
        "🔍 Compare Recordings",
        type="primary",
        disabled=(comp_rec_1 is None or comp_rec_2 is None)
    )

if compare_button:
    with st.spinner("🔍 Comparing recordings..."):
        try:
            rec1_data = st.session_state.multi_recordings[cell_key_1][comp_rec_1 - 1]
            rec2_data = st.session_state.multi_recordings[cell_key_2][comp_rec_2 - 1]
            compare_two_recordings(
                rec1_data, f"{comp_type_1} Recording #{comp_rec_1}",
                rec2_data, f"{comp_type_2} Recording #{comp_rec_2}"
            )
        except Exception as e:
            st.error(f"❌ Error during comparison: {str(e)}")
            with st.expander("🐛 See detailed error"):
                st.exception(e)

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# EXPORT SECTION
# =========================================================

st.markdown("---")
st.markdown("## 💾 Export Recordings")

export_col1, export_col2 = st.columns(2)

with export_col1:
    if len(st.session_state.multi_recordings["healthy"]) > 0:
        st.markdown("### Healthy Cell Recordings")
        for idx, recording in enumerate(st.session_state.multi_recordings["healthy"], 1):
            recording_data = recording.get('data', recording)
            if hasattr(recording_data, 'read'):
                recording_data = recording_data.read()
            metadata = recording.get('metadata', {})
            label = f"⬇️ Download Healthy #{idx}"
            if metadata:
                label += f" (Hit #{metadata.get('hit_number', 'N/A')})"
            st.download_button(
                label=label, data=recording_data,
                file_name=f"healthy_cell_recording_{idx}.wav",
                mime="audio/wav", key=f"download_healthy_{idx}"
            )

with export_col2:
    if len(st.session_state.multi_recordings["unhealthy"]) > 0:
        st.markdown("### Unhealthy Cell Recordings")
        for idx, recording in enumerate(st.session_state.multi_recordings["unhealthy"], 1):
            recording_data = recording.get('data', recording)
            if hasattr(recording_data, 'read'):
                recording_data = recording_data.read()
            metadata = recording.get('metadata', {})
            label = f"⬇️ Download Unhealthy #{idx}"
            if metadata:
                label += f" (Hit #{metadata.get('hit_number', 'N/A')})"
            st.download_button(
                label=label, data=recording_data,
                file_name=f"unhealthy_cell_recording_{idx}.wav",
                mime="audio/wav", key=f"download_unhealthy_{idx}"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; padding:1rem;">
    <p>Advanced Percussion Health Analysis Tool | Multi-Hit Detection &amp; Selection</p>
    <p>Record 5 hits → Review all hits → Select best hit → Analyze → Compare recordings</p>
</div>
""", unsafe_allow_html=True)
