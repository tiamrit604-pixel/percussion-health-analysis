import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
import pywt
import io

st.set_page_config(layout="wide", page_title="Percussion Health Analysis - Advanced")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hit-card {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        background-color: #f9f9f9;
    }
    .hit-card-selected {
        border: 3px solid #1f77b4;
        background-color: #e3f2fd;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎯 Percussion Health Analysis - Advanced</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Hit Recording & Selection System</p>', unsafe_allow_html=True)

# -----------------------------
# Session State Initialization
# -----------------------------
if "multi_recordings" not in st.session_state:
    st.session_state.multi_recordings = {
        "healthy": [],
        "unhealthy": []
    }

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
    
    # Advanced settings
    st.markdown("## 🎛️ Detection Settings")
    
    peak_threshold = st.slider(
        "Peak Detection Threshold",
        min_value=0.01,
        max_value=0.5,
        value=0.05,
        step=0.01,
        help="Minimum amplitude required to detect a hit"
    )
    
    min_hit_distance = st.slider(
        "Minimum Hit Distance (seconds)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Minimum time between consecutive hits"
    )
    
    hit_duration = st.slider(
        "Hit Duration (seconds)",
        min_value=0.2,
        max_value=1.0,
        value=0.5,
        step=0.1,
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
    
    ### **Tips for Best Results:**
    - 🎯 Hit the center of each cell consistently
    - ⏱️ Wait ~1 second between hits
    - 💪 Use consistent force for all hits
    - 🔇 Record in a quiet environment
    - 🎤 Keep microphone 10-15 cm from plate
    """)

# -----------------------------
# Hit Detection Function
# -----------------------------
def detect_multiple_hits(y, sr, height_threshold=0.05, min_distance=0.5, duration=0.5):

    # Absolute amplitude
    signal_energy = np.abs(y)

    # Smooth signal (removes micro spikes)
    signal_energy = np.convolve(signal_energy, np.ones(1000)/1000, mode='same')

    min_distance_samples = int(min_distance * sr)

    peaks, properties = find_peaks(
        signal_energy,
        height=height_threshold,
        distance=min_distance_samples
    )

    # ---- NEW: Keep only strong peaks ----
    if len(peaks) > 0:
        peak_heights = properties["peak_heights"]
        max_peak = np.max(peak_heights)

        # Keep peaks at least 40% of strongest hit
        strong_indices = peak_heights > 0.4 * max_peak
        peaks = peaks[strong_indices]
    # --------------------------------------

    hits = []
    duration_samples = int(duration * sr)

    for i, peak in enumerate(peaks):

        start = peak
        end = min(peak + duration_samples, len(y))
        hit_signal = y[start:end]

        if len(hit_signal) == 0:
            continue

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



# -----------------------------
# Main Recording Section
# -----------------------------
st.markdown(f"## 🎙️ Step 1: Record Multiple Hits - {cell_type} Cell")

col_rec1, col_rec2 = st.columns([2, 1])

with col_rec1:
    st.info(f"📍 Currently recording for: **{cell_type}** cell")
    st.warning("⚠️ **Important**: Record 5 consecutive hits with ~1 second pause between each hit")
    
    # Browser audio recording
    audio_bytes = st.audio_input(f"Record 5 hits for {cell_type} cell (click to start/stop)")
    
    if audio_bytes:
        st.success("✅ Audio captured! Click 'Process Recording' below to detect hits.")
        
        # Process button
        if st.button("🔍 Process Recording & Detect Hits", type="primary", key="process_btn"):
            try:
                # Read audio data
                audio_data = audio_bytes.read()
                audio_bytes.seek(0)
                
                # Load audio
                y, sr = librosa.load(io.BytesIO(audio_data), sr=None)
                
                if len(y) == 0:
                    st.error("❌ Recording is empty!")
                else:
                    # Detect hits
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
    st.markdown("### 📝 Quick Stats")
    if audio_bytes:
        audio_bytes_data = audio_bytes.read()
        audio_bytes.seek(0)
        st.info(f"Recording size: {len(audio_bytes_data)} bytes")
    else:
        st.warning("No recording yet")

# -----------------------------
# Display Detected Hits
# -----------------------------
if st.session_state.detected_hits is not None:
    st.markdown("---")
    st.markdown("## 👁️ Step 2: Review Detected Hits & Select Best One")
    
    hits_data = st.session_state.detected_hits
    hits = hits_data['hits']
    full_signal = hits_data['full_signal']
    sr = hits_data['sr']
    cell_type_recorded = hits_data['cell_type']
    
    st.info(f"📊 Found **{len(hits)} hit(s)** in the recording. Review each hit and select the best one for analysis.")
    
    # Display full signal with hit markers
    st.markdown("### 🎵 Full Recording with Detected Hits")
    
    fig_full, ax_full = plt.subplots(figsize=(14, 4))
    time_full = np.linspace(0, len(full_signal)/sr, len(full_signal))
    ax_full.plot(time_full, full_signal, linewidth=0.5, alpha=0.7, color='gray', label='Full Signal')
    
    # Mark each hit
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    for i, hit in enumerate(hits):
        onset_time = hit['onset_time']
        color = colors[i % len(colors)]
        ax_full.axvline(x=onset_time, color=color, linestyle='--', linewidth=2, 
                       label=f'Hit {i+1}', alpha=0.8)
        
        # Highlight the hit region
        hit_end_time = onset_time + hit['duration']
        ax_full.axvspan(onset_time, hit_end_time, alpha=0.2, color=color)
    
    ax_full.set_title(f"Full Recording - {cell_type_recorded} Cell ({len(hits)} hits detected)", 
                     fontsize=14, fontweight='bold')
    ax_full.set_xlabel("Time (s)")
    ax_full.set_ylabel("Amplitude")
    ax_full.legend(loc='upper right')
    ax_full.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_full)
    plt.close()
    
    # Display individual hits
    st.markdown("### 🎯 Individual Hit Waveforms")
    
    # Calculate grid layout
    n_hits = len(hits)
    n_cols = min(3, n_hits)
    n_rows = (n_hits + n_cols - 1) // n_cols
    
    # Create grid for hit display
    for row in range(n_rows):
        cols = st.columns(n_cols)
        
        for col_idx in range(n_cols):
            hit_idx = row * n_cols + col_idx
            
            if hit_idx < n_hits:
                hit = hits[hit_idx]
                
                with cols[col_idx]:
                    # Create a card for each hit
                    is_selected = (st.session_state.selected_hit_index == hit_idx)
                    card_class = "hit-card-selected" if is_selected else "hit-card"
                    
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    
                    st.markdown(f"#### Hit #{hit_idx + 1}")
                    
                    # Metrics
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Peak", f"{hit['peak_amplitude']:.3f}")
                    with metric_col2:
                        st.metric("RMS", f"{hit['rms']:.4f}")
                    
                    # Waveform
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
                    
                    # Selection button
                    button_label = "✅ Selected" if is_selected else f"Select Hit #{hit_idx + 1}"
                    button_type = "secondary" if is_selected else "primary"
                    
                    if st.button(button_label, key=f"select_hit_{hit_idx}", 
                               type=button_type, disabled=is_selected):
                        st.session_state.selected_hit_index = hit_idx
                        st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Save selected hit button
    st.markdown("---")
    if st.session_state.selected_hit_index is not None:
        selected_hit = hits[st.session_state.selected_hit_index]
        
        st.success(f"✅ Hit #{st.session_state.selected_hit_index + 1} selected!")
        
        col_save1, col_save2, col_save3 = st.columns([1, 1, 1])
        
        with col_save1:
            st.metric("Selected Hit", f"#{st.session_state.selected_hit_index + 1}")
        with col_save2:
            st.metric("Peak Amplitude", f"{selected_hit['peak_amplitude']:.3f}")
        with col_save3:
            st.metric("RMS Energy", f"{selected_hit['rms']:.4f}")
        
        if st.button("💾 Save Selected Hit & Continue", type="primary", key="save_hit"):
            # Save to recordings
            cell_key = cell_type_recorded.lower()
            
            # Create a simple audio buffer with the hit signal
            import io
            from scipy.io import wavfile
            
            # Save as WAV in memory
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
            
            # Clear detection data to allow new recording
            st.session_state.detected_hits = None
            st.session_state.selected_hit_index = None
    else:
        st.warning("⚠️ Please select a hit from the options above before continuing.")

# -----------------------------
# Analysis Section
# -----------------------------
st.markdown("---")
st.markdown("## 🔬 Step 3: Analyze Selected Hit")

analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

with analysis_col1:
    analyze_type = st.selectbox(
        "Select cell type to analyze:",
        ["Healthy", "Unhealthy"]
    )

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
    analyze_button = st.button("📊 Analyze Selected Recording", type="primary", 
                               disabled=(recording_index is None))

# Analysis function (same as before but enhanced)
def analyze_audio(y, sr, cell_type_label, metadata=None):
    """Comprehensive audio signal analysis"""
    
    st.markdown(f"### 🎯 Analysis Results - {cell_type_label} Cell")
    
    if metadata:
        st.info(f"📍 Analyzing Hit #{metadata.get('hit_number', 'N/A')} | "
               f"Peak: {metadata.get('peak_amplitude', 0):.3f} | "
               f"RMS: {metadata.get('rms', 0):.4f}")
    
    # Basic info
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Duration", f"{len(y)/sr:.2f} sec")
    with info_col2:
        st.metric("Sample Rate", f"{sr} Hz")
    with info_col3:
        st.metric("Samples", f"{len(y):,}")
    
    st.markdown("---")
    
    t = np.linspace(0, len(y)/sr, len(y))
    
    # TIME & FREQUENCY DOMAIN
    st.markdown("#### 📈 Time Domain & Frequency Domain")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # TIME DOMAIN
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t, y, linewidth=0.8, color='#1f77b4')
        ax1.set_title(f"Time Domain Waveform - {cell_type_label} Cell", 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel("Time (s)", fontsize=11)
        ax1.set_ylabel("Amplitude", fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        
        # FFT
        fft = np.fft.fft(y)
        freq = np.fft.fftfreq(len(fft), 1/sr)
        positive_mask = freq > 0
        freq_positive = freq[positive_mask]
        fft_magnitude = np.abs(fft[positive_mask])
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(freq_positive, fft_magnitude, linewidth=0.8, color='#ff7f0e')
        ax2.set_title(f"FFT Spectrum - {cell_type_label} Cell", 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel("Frequency (Hz)", fontsize=11)
        ax2.set_ylabel("Magnitude", fontsize=11)
        ax2.set_xlim([0, min(10000, sr/2)])
        ax2.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        
        peak_freq = freq_positive[np.argmax(fft_magnitude)]
        st.info(f"🎵 **Dominant Frequency**: {peak_freq:.1f} Hz")
    
    with col2:
        # PSD
        f_psd, psd = welch(y, sr, nperseg=min(256, len(y)//4))
        
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.semilogy(f_psd, psd, linewidth=1.0, color='#2ca02c')
        ax3.set_title(f"Power Spectral Density - {cell_type_label} Cell", 
                     fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel("Frequency (Hz)", fontsize=11)
        ax3.set_ylabel("Power/Frequency (dB/Hz)", fontsize=11)
        ax3.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        frames = range(len(spectral_centroids))
        time_centroid = librosa.frames_to_time(frames, sr=sr)
        
        fig_sc, ax_sc = plt.subplots(figsize=(10, 4))
        ax_sc.plot(time_centroid, spectral_centroids, linewidth=1.5, 
                  color='#d62728', marker='o', markersize=3)
        ax_sc.set_title(f"Spectral Centroid - {cell_type_label} Cell", 
                       fontsize=14, fontweight='bold', pad=15)
        ax_sc.set_xlabel("Time (s)", fontsize=11)
        ax_sc.set_ylabel("Frequency (Hz)", fontsize=11)
        ax_sc.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig_sc)
        plt.close()
        
        st.info(f"🎼 **Mean Spectral Centroid**: {np.mean(spectral_centroids):.1f} Hz")
    
    # TIME-FREQUENCY DOMAIN
    st.markdown("---")
    st.markdown("#### 🌊 Time-Frequency Domain")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # STFT
        D = librosa.stft(y, n_fft=2048, hop_length=512)
        DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz', 
                                      ax=ax4, cmap='viridis')
        fig4.colorbar(img, ax=ax4, format='%+2.0f dB')
        ax4.set_title(f"STFT Spectrogram - {cell_type_label} Cell", 
                     fontsize=14, fontweight='bold', pad=15)
        ax4.set_ylim([0, min(10000, sr/2)])
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
        
        # CWT
        scales = np.arange(1, 128)
        coeffs, frequencies = pywt.cwt(y, scales, 'morl', sampling_period=1/sr)
        
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        im = ax5.imshow(np.abs(coeffs), aspect='auto', cmap='jet', 
                       extent=[0, len(y)/sr, frequencies[-1], frequencies[0]])
        ax5.set_title(f"CWT Scalogram - {cell_type_label} Cell", 
                     fontsize=14, fontweight='bold', pad=15)
        ax5.set_xlabel("Time (s)", fontsize=11)
        ax5.set_ylabel("Frequency (Hz)", fontsize=11)
        fig5.colorbar(im, ax=ax5, label='Magnitude')
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()
    
    with col4:
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        img2 = librosa.display.specshow(mfcc.T, sr=sr, x_axis='time', ax=ax6, cmap='coolwarm')
        fig6.colorbar(img2, ax=ax6, label='MFCC Coefficient')
        ax6.set_title(f"MFCC - {cell_type_label} Cell", 
                     fontsize=14, fontweight='bold', pad=15)
        ax6.set_ylabel("MFCC Coefficients", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()
        
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        img3 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', 
                                       y_axis='mel', ax=ax7, cmap='magma')
        fig7.colorbar(img3, ax=ax7, format='%+2.0f dB')
        ax7.set_title(f"Mel Spectrogram - {cell_type_label} Cell", 
                     fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()
    
    # Feature Summary
    st.markdown("---")
    st.markdown("#### 📊 Feature Summary")
    
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    rms_energy = librosa.feature.rms(y=y)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Mean Zero Crossing Rate", f"{np.mean(zero_crossing_rate):.4f}")
    with metric_col2:
        st.metric("Mean RMS Energy", f"{np.mean(rms_energy):.4f}")
    with metric_col3:
        st.metric("Mean Spectral Centroid", f"{np.mean(spectral_centroids):.1f} Hz")
    with metric_col4:
        st.metric("Mean Spectral Rolloff", f"{np.mean(spectral_rolloff):.1f} Hz")

# Execute Analysis
if analyze_button and recording_index is not None:
    cell_key_analyze = analyze_type.lower()
    selected_recording = st.session_state.multi_recordings[cell_key_analyze][recording_index - 1]
    
    with st.spinner(f"🔍 Analyzing {analyze_type} cell recording #{recording_index}..."):
        try:
            # Get signal and metadata
            if 'signal' in selected_recording:
                y = selected_recording['signal']
                sr = selected_recording['sr']
                metadata = selected_recording.get('metadata', None)
            else:
                # Fallback for old format
                audio_data = selected_recording.get('data', selected_recording)
                if hasattr(audio_data, 'read'):
                    audio_data = audio_data.read()
                    audio_data.seek(0) if hasattr(audio_data, 'seek') else None
                
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

# Export Section
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
                recording_data.seek(0) if hasattr(recording_data, 'seek') else None
            
            metadata = recording.get('metadata', {})
            label = f"⬇️ Download Healthy #{idx}"
            if metadata:
                label += f" (Hit #{metadata.get('hit_number', 'N/A')})"
            
            st.download_button(
                label=label,
                data=recording_data,
                file_name=f"healthy_cell_recording_{idx}.wav",
                mime="audio/wav",
                key=f"download_healthy_{idx}"
            )

with export_col2:
    if len(st.session_state.multi_recordings["unhealthy"]) > 0:
        st.markdown("### Unhealthy Cell Recordings")
        for idx, recording in enumerate(st.session_state.multi_recordings["unhealthy"], 1):
            recording_data = recording.get('data', recording)
            if hasattr(recording_data, 'read'):
                recording_data = recording_data.read()
                recording_data.seek(0) if hasattr(recording_data, 'seek') else None
            
            metadata = recording.get('metadata', {})
            label = f"⬇️ Download Unhealthy #{idx}"
            if metadata:
                label += f" (Hit #{metadata.get('hit_number', 'N/A')})"
            
            st.download_button(
                label=label,
                data=recording_data,
                file_name=f"unhealthy_cell_recording_{idx}.wav",
                mime="audio/wav",
                key=f"download_unhealthy_{idx}"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p><strong>Advanced Percussion Health Analysis Tool</strong> | Multi-Hit Detection & Selection</p>
    <p>Record 5 hits → Review all hits → Select best hit → Analyze</p>
</div>
""", unsafe_allow_html=True)
