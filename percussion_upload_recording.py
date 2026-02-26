import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import pywt
from pydub import AudioSegment
from scipy.signal import find_peaks, stft, periodogram
from scipy.fft import fft, fftfreq

# Specific SPAFE imports as per your logic
try:
    from spafe.features import mfcc
    from spafe.utils.preprocessing import SlidingWindow
    HAS_SPAFE = True
except ImportError:
    HAS_SPAFE = False

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Signal Processing Analysis for Percussion Signal")

# --- Custom CSS (Glassmorphism UI) ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 15px; margin-bottom: 1.5rem; text-align: center;
    }
    .main-title { color: #e94560; font-size: 2.5rem; font-weight: 800; }
    .main-subtitle { color: #a8b2d8; font-size: 1.1rem; }
    .hit-card {
        border: 1px solid #374151; border-radius: 12px; padding: 15px;
        background: #1f2937; margin-bottom: 10px;
    }
    .hit-card-selected {
        border: 2px solid #00ff88; border-radius: 12px; padding: 15px;
        background: rgba(0, 255, 136, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "multi_recordings" not in st.session_state:
    st.session_state.multi_recordings = {"healthy": [], "unhealthy": []}
if "detected_hits" not in st.session_state:
    st.session_state.detected_hits = None
if "selected_hit_index" not in st.session_state:
    st.session_state.selected_hit_index = None

# =========================================================
# SIGNAL PROCESSING HELPERS (Notebook Logic)
# =========================================================

def load_audio_any_format(file_bytes):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
        return y, sr
    except:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        sr = audio.frame_rate
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels == 2: samples = samples.reshape((-1, 2)).mean(axis=1)
        samples /= np.max(np.abs(samples)) if np.max(np.abs(samples)) > 0 else 1
        return samples, sr

def detect_multiple_hits(y, sr, height_threshold=0.05, min_distance=0.5, duration=0.2):
    abs_y = np.abs(y)
    effective_height = max(height_threshold, 0.3 * np.max(abs_y))
    peaks, _ = find_peaks(abs_y, height=effective_height, distance=int(min_distance * sr))
    hits = []
    pre_samples, duration_samples = int(0.02 * sr), int(duration * sr)
    for i, peak in enumerate(peaks):
        start, end = max(0, peak - pre_samples), min(len(y), peak + duration_samples)
        hit_signal = y[start:end]
        if len(hit_signal) == 0: continue
        max_val = np.max(np.abs(hit_signal))
        hit_signal = hit_signal / max_val if max_val > 0 else hit_signal
        hits.append({
            'index': i, 'onset_time': peak / sr, 'signal': hit_signal,
            'rms': np.sqrt(np.mean(hit_signal**2)), 'peak': max_val, 'duration': len(hit_signal)/sr
        })
    return hits

# =========================================================
# PLOTTING FUNCTIONS (Exact Notebook Implementation)
# =========================================================

def _plot_time_domain(ax, y, sr, title, color='#1f77b4'):
    time = np.linspace(0, len(y)/sr, len(y))
    ax.plot(time, y, color=color, lw=1)
    ax.set_title(title, fontweight='bold'); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

def _plot_fft(ax, y, sr, title, color='#ff7f0e'):
    N = len(y)
    yf = fft(y.astype(np.float64))
    P2 = np.abs(yf / N)
    P1 = P2[:N // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    xf = fftfreq(N, 1 / sr)[:N // 2 + 1]
    peak_freq = xf[np.argmax(P1)]
    ax.plot(xf, P1, color=color, lw=1)
    ax.set_xlim(0, 10000); ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("Amplitude"); ax.grid(True, alpha=0.3)
    return peak_freq

def _plot_stft(fig, ax, y, sr, title):
    f, t, Zxx = stft(y.astype(np.float64), sr, nperseg=256, noverlap=200, nfft=512)
    pcm = ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
    ax.set_ylim(0, 10000); ax.set_title(title, fontweight='bold')
    fig.colorbar(pcm, ax=ax, label='Magnitude')

def _plot_psd(ax, y, sr, title, color='#2ca02c'):
    f, pxx = periodogram(y.astype(np.float64), fs=1.0, window='boxcar', nfft=4096)
    ax.plot(f, pxx, color=color, lw=1.2)
    ax.set_title(title, fontweight='bold'); ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Power/Frequency"); ax.set_xlim(0, 0.5); ax.grid(True, alpha=0.3)

def _plot_cwt(fig, ax, y, sr, title):
    scales = np.arange(1, 256)
    coeffs, freqs = pywt.cwt(y.astype(np.float64), scales, 'morl', sampling_period=1/sr)
    im = ax.imshow(np.abs(coeffs), extent=[0, len(y)/sr, freqs[-1], freqs[0]], aspect='auto', cmap='jet')
    ax.set_title(title, fontweight='bold'); fig.colorbar(im, ax=ax)

def _plot_mfcc(fig, ax, y, sr, title):
    if HAS_SPAFE:
        mfccs = mfcc.imfcc(y.astype(np.float64), fs=sr, num_ceps=13, 
                          window=SlidingWindow(win_len=0.025, win_hop=0.01), nfilts=128, nfft=2048, low_freq=200, high_freq=15000)
        im = ax.imshow(mfccs.T, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(title, fontweight='bold'); fig.colorbar(im, ax=ax)

# =========================================================
# MAIN APP
# =========================================================

st.markdown('<div class="main-header"><div class="main-title"> Automation of Signal Processing Analysis for Percussion Signal</div><div class="main-subtitle">Multi-Hit Selection & Detailed Signal Diagnostics</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    cell_mode = st.radio("Recording Category:", ["Healthy", "Unhealthy"])
    peak_thr = st.slider("Trigger Sensitivity", 0.01, 0.5, 0.2)
    hit_win = st.slider("Hit Window (s)", 0.1, 1.0, 0.2) # Default 0.2
    st.divider()
    if st.button("🗑️ Reset Session"):
        st.session_state.multi_recordings = {"healthy": [], "unhealthy": []}
        st.rerun()

# --- Step 1: Upload ---
st.header(f"Step 1: Process {cell_mode} File")
up_file = st.file_uploader("Upload Audio (Multiple Hits)", type=["wav", "mp3", "m4a","mp4"])

if up_file and st.button("🔍 Scan Hits", type="primary"):
    y, sr = load_audio_any_format(up_file.read())
    hits = detect_multiple_hits(y, sr, height_threshold=peak_thr, duration=hit_win)
    if hits:
        st.session_state.detected_hits = {'hits': hits, 'sr': sr, 'type': cell_mode}
        st.success(f"Isolated {len(hits)} impacts.")
    else: st.error("No hits detected.")

# --- Step 2: Hit Selection ---
if st.session_state.detected_hits:
    st.divider()
    st.header("Step 2: Select Best Hit")
    d_hits = st.session_state.detected_hits
    cols = st.columns(3)
    for i, h in enumerate(d_hits['hits']):
        with cols[i % 3]:
            is_sel = st.session_state.selected_hit_index == i
            st.markdown(f'<div class="{"hit-card-selected" if is_sel else "hit-card"}">', unsafe_allow_html=True)
            st.write(f"**Impact #{i+1}** ({h['onset_time']:.2f}s)")
            fig_h, ax_h = plt.subplots(figsize=(4, 2))
            ax_h.plot(h['signal'], color='#00ff88', lw=1); ax_h.axis('off'); fig_h.patch.set_facecolor('#1f2937')
            st.pyplot(fig_h)
            if st.button(f"Use #{i+1}", key=f"sel_{i}"):
                st.session_state.selected_hit_index = i
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.selected_hit_index is not None:
        if st.button("💾 Save & Open Diagnostics", type="primary", use_container_width=True):
            h = d_hits['hits'][st.session_state.selected_hit_index]
            st.session_state.multi_recordings[d_hits['type'].lower()].append({
                'signal': h['signal'], 'sr': d_hits['sr'], 'label': f"{d_hits['type']} Sample {len(st.session_state.multi_recordings[d_hits['type'].lower()])+1}"
            })
            st.session_state.detected_hits = None; st.session_state.selected_hit_index = None
            st.rerun()

# --- Step 3: View Individual / Comparison ---
if any(st.session_state.multi_recordings.values()):
    st.divider()
    tab_single, tab_compare = st.tabs(["📊 Individual Analysis", "🔄 Compare Samples"])
    
    with tab_single:
        v_cat = st.selectbox("View Category", ["Healthy", "Unhealthy"])
        v_list = st.session_state.multi_recordings[v_cat.lower()]
        if v_list:
            v_idx = st.selectbox("Select Recording", range(len(v_list)), format_func=lambda x: v_list[x]['label'])
            sel = v_list[v_idx]
            y, sr = sel['signal'], sel['sr']
            
            # Grid of all 6 plots
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(); _plot_time_domain(ax, y, sr, "Time Domain"); st.pyplot(fig)
                fig, ax = plt.subplots(); _plot_fft(ax, y, sr, "FFT Spectrum"); st.pyplot(fig)
                fig, ax = plt.subplots(); _plot_psd(ax, y, sr, "Power Spectral Density"); st.pyplot(fig)
            with c2:
                fig, ax = plt.subplots(); _plot_stft(fig, ax, y, sr, "STFT Spectrogram"); st.pyplot(fig)
                fig, ax = plt.subplots(); _plot_cwt(fig, ax, y, sr, "CWT Scalogram"); st.pyplot(fig)
                fig, ax = plt.subplots(); _plot_mfcc(fig, ax, y, sr, "MFCC Heatmap"); st.pyplot(fig)
        else: st.info("No recordings saved here.")

    with tab_compare:
        colL, colR = st.columns(2)
        with colL:
            st1 = st.selectbox("Sample 1 Type", ["Healthy", "Unhealthy"], key="st1")
            rec1 = st.session_state.multi_recordings[st1.lower()]
            id1 = st.selectbox("Sample 1 ID", range(len(rec1)), format_func=lambda x: rec1[x]['label'], key="id1") if rec1 else None
        with colR:
            st2 = st.selectbox("Sample 2 Type", ["Healthy", "Unhealthy"], key="st2")
            rec2 = st.session_state.multi_recordings[st2.lower()]
            id2 = st.selectbox("Sample 2 ID", range(len(rec2)), format_func=lambda x: rec2[x]['label'], key="id2") if rec2 else None
        
        if id1 is not None and id2 is not None:
            r1, r2 = rec1[id1], rec2[id2]
            for mode in ["Time Domain", "FFT Spectrum", "STFT Spectrogram", "PSD", "CWT Scalogram", "MFCC Heatmap"]:
                st.subheader(f"🔄 {mode} Comparison")
                fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 4))
                if mode == "Time Domain":
                    _plot_time_domain(axL, r1['signal'], r1['sr'], r1['label'], '#00ff88')
                    _plot_time_domain(axR, r2['signal'], r2['sr'], r2['label'], '#f43f5e')
                elif mode == "FFT Spectrum":
                    _plot_fft(axL, r1['signal'], r1['sr'], r1['label'], '#00ff88')
                    _plot_fft(axR, r2['signal'], r2['sr'], r2['label'], '#f43f5e')
                elif mode == "STFT Spectrogram":
                    _plot_stft(fig, axL, r1['signal'], r1['sr'], r1['label'])
                    _plot_stft(fig, axR, r2['signal'], r2['sr'], r2['label'])
                elif mode == "PSD":
                    _plot_psd(axL, r1['signal'], r1['sr'], r1['label'], '#00ff88')
                    _plot_psd(axR, r2['signal'], r2['sr'], r2['label'], '#f43f5e')
                elif mode == "CWT Scalogram":
                    _plot_cwt(fig, axL, r1['signal'], r1['sr'], r1['label'])
                    _plot_cwt(fig, axR, r2['signal'], r2['sr'], r2['label'])
                elif mode == "MFCC Heatmap":
                    _plot_mfcc(fig, axL, r1['signal'], r1['sr'], r1['label'])
                    _plot_mfcc(fig, axR, r2['signal'], r2['sr'], r2['label'])
                st.pyplot(fig); plt.close()



# --- Footer Copyright ---
st.markdown(
    """
    <hr style="margin-top:50px;">
    <div style='text-align:center; color:gray; font-size:14px;'>
        © 2026 Amrit Tiwari | Automation of Signal Processing Analysis for Percussion Signal | University of Houston
    </div>
    """,
    unsafe_allow_html=True
)
