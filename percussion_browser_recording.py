import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import welch
import pywt
import io
from scipy.io import wavfile

st.set_page_config(layout="wide", page_title="Percussion Health Analysis")

# Custom CSS for better UI
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
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎯 Percussion Health Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Composite Plate Health Monitoring System</p>', unsafe_allow_html=True)

# -----------------------------
# Session State Initialization
# -----------------------------
if "recordings" not in st.session_state:
    st.session_state.recordings = {
        "healthy": [],
        "unhealthy": []
    }

# -----------------------------
# Instructions
# -----------------------------
with st.expander("📋 How to Use This Tool", expanded=True):
    st.markdown("""
    ### Step-by-Step Instructions:
    
    1. **Select Cell Type** (Healthy or Unhealthy)
    2. **Click the microphone button** to start recording
    3. **Hit the plate** with consistent force
    4. **Stop recording** after the hit
    5. **Click "Analyze Latest Recording"** to see results
    6. **Repeat 5 times** per cell type for best results
    
    ### Tips for Best Results:
    - 🎤 Allow microphone access when prompted
    - 🔇 Record in a quiet environment
    - 💪 Use consistent striking force
    - 📏 Keep microphone distance constant (10-15 cm from plate)
    - 🎯 Hit the center of each cell
    - ⏱️ Record for 2-3 seconds per hit
    """)

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
    
    st.markdown("---")
    
    st.markdown("## 📊 Recording Stats")
    st.metric("Healthy Recordings", len(st.session_state.recordings["healthy"]))
    st.metric("Unhealthy Recordings", len(st.session_state.recordings["unhealthy"]))
    
    st.markdown("---")
    
    if st.button("🗑️ Clear All Recordings"):
        st.session_state.recordings = {
            "healthy": [],
            "unhealthy": []
        }
        st.success("All recordings cleared!")
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("## 📖 About")
    st.markdown("""
    This tool analyzes percussion signals to detect:
    - **Healthy cells**: Good bonding with epoxy
    - **Unhealthy cells**: Debonding, no epoxy
    
    ### Analysis Methods:
    - **Time Domain**: Waveform
    - **FFT**: Frequency spectrum
    - **PSD**: Power distribution
    - **STFT**: Spectrogram
    - **CWT**: Wavelet transform
    - **MFCC**: Cepstral coefficients
    """)

# -----------------------------
# Main Recording Section
# -----------------------------
st.markdown(f"## 🎙️ Recording - {cell_type} Cell")

col_rec1, col_rec2 = st.columns([2, 1])

with col_rec1:
    st.info(f"📍 Currently recording for: **{cell_type}** cell")
    
    # Browser audio recording
    audio_bytes = st.audio_input(f"Record percussion sound for {cell_type} cell")
    
    if audio_bytes:
        st.success("✅ Audio captured! Click 'Save & Analyze' below.")
        
        # Save button
        if st.button("💾 Save & Analyze This Recording", type="primary"):
            # Read the audio data
            audio_data = audio_bytes.read()
            audio_bytes.seek(0)  # Reset for potential later use
            
            # Store in session state
            cell_key = cell_type.lower()
            st.session_state.recordings[cell_key].append(audio_data)
            st.success(f"✅ Recording #{len(st.session_state.recordings[cell_key])} saved for {cell_type} cell!")
            st.balloons()

with col_rec2:
    st.markdown("### 📝 Quick Stats")
    if audio_bytes:
        # Get size from UploadedFile object
        audio_bytes_data = audio_bytes.read()
        audio_bytes.seek(0)  # Reset position for later use
        st.info(f"Recording size: {len(audio_bytes_data)} bytes")
    else:
        st.warning("No recording yet")

# -----------------------------
# Analysis Section
# -----------------------------
st.markdown("---")
st.markdown("## 🔬 Signal Analysis")

# Analysis controls
analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

with analysis_col1:
    analyze_type = st.selectbox(
        "Select cell type to analyze:",
        ["Healthy", "Unhealthy"]
    )

with analysis_col2:
    cell_key_analyze = analyze_type.lower()
    num_recordings = len(st.session_state.recordings[cell_key_analyze])
    
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
    st.write("")  # Spacing
    st.write("")  # Spacing
    analyze_button = st.button("📊 Analyze Selected Recording", type="primary", disabled=(recording_index is None))

# -----------------------------
# Analysis Function
# -----------------------------
def analyze_audio(y, sr, cell_type_label):
    """
    Comprehensive audio signal analysis
    """
    
    st.markdown(f"### 🎯 Analysis Results - {cell_type_label} Cell")
    
    # Basic info
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Duration", f"{len(y)/sr:.2f} sec")
    with info_col2:
        st.metric("Sample Rate", f"{sr} Hz")
    with info_col3:
        st.metric("Samples", f"{len(y):,}")
    
    st.markdown("---")
    
    # -----------------------------
    # Onset Detection & Segmentation
    # -----------------------------
    st.markdown("#### 🎯 Signal Preprocessing")
    
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='samples')
    
    if len(onset_frames) > 0:
        onset = onset_frames[0]
        duration_samples = int(0.5 * sr)
        y_processed = y[onset:onset + duration_samples]
        st.success(f"✂️ Onset detected at {onset/sr:.3f}s. Extracted {len(y_processed)/sr:.3f}s segment.")
    else:
        duration_samples = int(0.5 * sr)
        y_processed = y[:duration_samples]
        st.warning("⚠️ No clear onset detected. Using first 0.5 seconds.")
    
    if len(y_processed) < 100:
        st.error("❌ Signal too short after processing.")
        return
    
    t = np.linspace(0, len(y_processed)/sr, len(y_processed))
    
    # -----------------------------
    # TIME & FREQUENCY DOMAIN
    # -----------------------------
    st.markdown("#### 📈 Time Domain & Frequency Domain")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # TIME DOMAIN
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t, y_processed, linewidth=0.8, color='#1f77b4')
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
        fft = np.fft.fft(y_processed)
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
        
        # Dominant frequency
        peak_freq = freq_positive[np.argmax(fft_magnitude)]
        st.info(f"🎵 **Dominant Frequency**: {peak_freq:.1f} Hz")
    
    with col2:
        # PSD
        f_psd, psd = welch(y_processed, sr, nperseg=min(256, len(y_processed)//4))
        
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
        spectral_centroids = librosa.feature.spectral_centroid(y=y_processed, sr=sr)[0]
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
    
    # -----------------------------
    # TIME-FREQUENCY DOMAIN
    # -----------------------------
    st.markdown("---")
    st.markdown("#### 🌊 Time-Frequency Domain")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # STFT
        D = librosa.stft(y_processed, n_fft=2048, hop_length=512)
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
        coeffs, frequencies = pywt.cwt(y_processed, scales, 'morl', sampling_period=1/sr)
        
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        im = ax5.imshow(np.abs(coeffs), aspect='auto', cmap='jet', 
                       extent=[0, len(y_processed)/sr, frequencies[-1], frequencies[0]])
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
        mfcc = librosa.feature.mfcc(y=y_processed, sr=sr, n_mfcc=13)
        
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        img2 = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax6, cmap='coolwarm')
        fig6.colorbar(img2, ax=ax6, label='MFCC Coefficient')
        ax6.set_title(f"MFCC - {cell_type_label} Cell", 
                     fontsize=14, fontweight='bold', pad=15)
        ax6.set_ylabel("MFCC Coefficients", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()
        
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y_processed, sr=sr, n_mels=128)
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
    
    # -----------------------------
    # Feature Summary
    # -----------------------------
    st.markdown("---")
    st.markdown("#### 📊 Feature Summary")
    
    # Calculate features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y_processed)[0]
    rms_energy = librosa.feature.rms(y=y_processed)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_processed, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_processed, sr=sr)[0]
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Mean Zero Crossing Rate", f"{np.mean(zero_crossing_rate):.4f}")
    with metric_col2:
        st.metric("Mean RMS Energy", f"{np.mean(rms_energy):.4f}")
    with metric_col3:
        st.metric("Mean Spectral Centroid", f"{np.mean(spectral_centroids):.1f} Hz")
    with metric_col4:
        st.metric("Mean Spectral Rolloff", f"{np.mean(spectral_rolloff):.1f} Hz")
    
    # Detailed statistics
    with st.expander("📈 Detailed Statistics & Features"):
        st.markdown("##### Signal Statistics")
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.write(f"- **Duration**: {len(y_processed)/sr:.3f} seconds")
            st.write(f"- **Sample Rate**: {sr} Hz")
            st.write(f"- **Total Samples**: {len(y_processed):,}")
            st.write(f"- **Max Amplitude**: {np.max(np.abs(y_processed)):.4f}")
            st.write(f"- **RMS Amplitude**: {np.sqrt(np.mean(y_processed**2)):.4f}")
        
        with stats_col2:
            st.write(f"- **Peak Frequency**: {peak_freq:.1f} Hz")
            st.write(f"- **Spectral Centroid**: {np.mean(spectral_centroids):.1f} Hz")
            st.write(f"- **Spectral Bandwidth**: {np.mean(spectral_bandwidth):.1f} Hz")
            st.write(f"- **Spectral Rolloff**: {np.mean(spectral_rolloff):.1f} Hz")
            st.write(f"- **Zero Crossing Rate**: {np.mean(zero_crossing_rate):.4f}")
        
        st.markdown("##### MFCC Statistics")
        mfcc_data = []
        for i in range(min(13, mfcc.shape[0])):
            mfcc_data.append({
                "Coefficient": f"MFCC {i+1}",
                "Mean": f"{np.mean(mfcc[i]):.3f}",
                "Std Dev": f"{np.std(mfcc[i]):.3f}",
                "Min": f"{np.min(mfcc[i]):.3f}",
                "Max": f"{np.max(mfcc[i]):.3f}"
            })
        
        import pandas as pd
        st.dataframe(pd.DataFrame(mfcc_data), use_container_width=True)

# -----------------------------
# Execute Analysis
# -----------------------------
if analyze_button and recording_index is not None:
    cell_key_analyze = analyze_type.lower()
    selected_recording = st.session_state.recordings[cell_key_analyze][recording_index - 1]
    
    with st.spinner(f"🔍 Analyzing {analyze_type} cell recording #{recording_index}..."):
        try:
            # Handle both bytes and UploadedFile objects
            if hasattr(selected_recording, 'read'):
                # It's an UploadedFile object
                audio_data = selected_recording.read()
                selected_recording.seek(0)
            else:
                # It's already bytes
                audio_data = selected_recording
            
            # Load audio from bytes
            y, sr = librosa.load(io.BytesIO(audio_data), sr=None)
            
            # Check if audio was loaded successfully
            if len(y) == 0:
                st.error("❌ Recording is empty. Please record again.")
            else:
                # Run analysis
                analyze_audio(y, sr, analyze_type)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.info("💡 **Troubleshooting tips:**")
            st.write("- Make sure you recorded audio (hit the plate)")
            st.write("- Try recording again with a clearer impact")
            st.write("- Check that your microphone is working")
            st.write("- Record for at least 1-2 seconds")
            with st.expander("🐛 See detailed error"):
                st.exception(e)

# -----------------------------
# Download Section
# -----------------------------
st.markdown("---")
st.markdown("## 💾 Export Recordings")

export_col1, export_col2 = st.columns(2)

with export_col1:
    if len(st.session_state.recordings["healthy"]) > 0:
        st.markdown("### Healthy Cell Recordings")
        for idx, recording in enumerate(st.session_state.recordings["healthy"], 1):
            # Handle both bytes and UploadedFile objects
            if hasattr(recording, 'read'):
                recording_data = recording.read()
                recording.seek(0)
            else:
                recording_data = recording
                
            st.download_button(
                label=f"⬇️ Download Healthy Recording #{idx}",
                data=recording_data,
                file_name=f"healthy_cell_recording_{idx}.wav",
                mime="audio/wav",
                key=f"download_healthy_{idx}"
            )

with export_col2:
    if len(st.session_state.recordings["unhealthy"]) > 0:
        st.markdown("### Unhealthy Cell Recordings")
        for idx, recording in enumerate(st.session_state.recordings["unhealthy"], 1):
            # Handle both bytes and UploadedFile objects
            if hasattr(recording, 'read'):
                recording_data = recording.read()
                recording.seek(0)
            else:
                recording_data = recording
                
            st.download_button(
                label=f"⬇️ Download Unhealthy Recording #{idx}",
                data=recording_data,
                file_name=f"unhealthy_cell_recording_{idx}.wav",
                mime="audio/wav",
                key=f"download_unhealthy_{idx}"
            )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p><strong>Percussion Health Analysis Tool</strong> | Composite Plate Testing</p>
    <p>Make sure to record 5 hits per cell type for comprehensive analysis</p>
</div>
""", unsafe_allow_html=True)