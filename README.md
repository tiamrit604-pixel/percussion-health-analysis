# 🎯 Percussion Health Analysis - Browser Recording Version

Complete web application for analyzing percussion signals from composite plates using browser-based audio recording.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run percussion_browser_recording.py
```

### 3. Access the App

The app will automatically open in your browser at `http://localhost:8501`

## 📋 How to Use

### Step 1: Grant Microphone Permission
- When you first click the record button, your browser will ask for microphone permission
- Click "Allow" to enable audio recording

### Step 2: Record Percussion Sounds

**For HEALTHY cells:**
1. Select "Healthy" in the sidebar
2. Click the microphone button
3. Hit the healthy cell on the plate
4. Stop recording (automatically stops after a few seconds)
5. Click "Save & Analyze This Recording"
6. Repeat 5 times for the same cell

**For UNHEALTHY cells:**
1. Select "Unhealthy" in the sidebar
2. Click the microphone button
3. Hit the unhealthy cell on the plate
4. Stop recording
5. Click "Save & Analyze This Recording"
6. Repeat 5 times for the same cell

### Step 3: Analyze Recordings

1. Select the cell type you want to analyze (Healthy or Unhealthy)
2. Choose which recording number (1-5)
3. Click "📊 Analyze Selected Recording"
4. View all the signal processing results

### Step 4: Export Results

- Download individual recordings using the download buttons at the bottom
- Take screenshots of the analysis plots for your report
- Compare healthy vs unhealthy results

## 📊 Analysis Features

The app provides comprehensive signal analysis:

### Time Domain Analysis
- **Waveform**: Raw signal amplitude over time
- **Onset Detection**: Automatically finds the percussion impact

### Frequency Domain Analysis
- **FFT**: Fast Fourier Transform showing frequency components
- **PSD**: Power Spectral Density
- **Spectral Centroid**: Center of mass of the spectrum

### Time-Frequency Domain Analysis
- **STFT**: Short-Time Fourier Transform (Spectrogram)
- **CWT**: Continuous Wavelet Transform (Scalogram)
- **MFCC**: Mel-Frequency Cepstral Coefficients
- **Mel Spectrogram**: Mel-scale frequency representation

### Feature Extraction
- Zero Crossing Rate
- RMS Energy
- Spectral Rolloff
- Spectral Bandwidth
- Detailed MFCC statistics

## 🎓 For Your Lab Report

### Data Collection
1. Record 5 hits for each cell type (healthy and unhealthy)
2. Download all recordings for documentation
3. Take photos of your experimental setup

### Analysis
1. Select the best recording from each set of 5
2. Run analysis on both (one healthy, one unhealthy)
3. Take screenshots of all plots
4. Note the key differences in features

### Key Observations to Report

**Healthy vs Unhealthy Cells:**
- Compare peak frequencies
- Compare signal decay rates
- Compare spectral centroids
- Compare MFCC patterns
- Note differences in time-frequency representations

## 🔧 Troubleshooting

### Microphone Not Working
- Check browser permissions (usually shows in address bar)
- Try refreshing the page
- Use Chrome or Edge for best compatibility
- Check your OS microphone settings

### Recording Quality Issues
- Get closer to the plate (10-15 cm optimal)
- Reduce background noise
- Hit the plate firmly and consistently
- Make sure the plate is on a stable surface

### Analysis Errors
- Ensure recordings are at least 0.5 seconds
- Check that you hit the plate (signal should show clear onset)
- Try recording again if onset detection fails

## 💡 Best Practices

1. **Consistent Setup**
   - Same microphone distance for all recordings
   - Same striking tool and force
   - Same position on each cell

2. **Recording Quality**
   - Quiet environment
   - Stable plate placement
   - Clear, audible impacts

3. **Data Management**
   - Record 5 hits per cell
   - Label your data clearly
   - Download recordings immediately after collection

4. **Analysis**
   - Compare similar recording numbers (e.g., Healthy #3 vs Unhealthy #3)
   - Look for consistent patterns across multiple recordings
   - Note any outliers

## 📝 Report Writing Tips

### Include in Your Report:

1. **Experimental Setup**
   - Photo of the plate with cells marked
   - Photos of team performing percussion tests
   - Description of recording setup

2. **Signal Processing Results**
   - All 6 plots for both healthy and unhealthy cells
   - Side-by-side comparisons
   - Feature summary tables

3. **Observations**
   - Time domain differences (decay rate, amplitude)
   - Frequency domain differences (peak frequencies, bandwidth)
   - Time-frequency differences (energy distribution patterns)

4. **Discussion**
   - Why healthy and unhealthy cells sound different
   - Which analysis method is most discriminative
   - Limitations of your dataset (sample size, environmental factors)

## 🎯 Expected Differences

**Healthy Cells (with epoxy bonding):**
- Higher frequency components
- Shorter, sharper impacts
- More concentrated energy in specific frequency bands
- Clearer harmonic structure

**Unhealthy Cells (debonded):**
- Lower dominant frequencies
- Longer decay times (more "ringing")
- Broader frequency distribution
- Different spectral centroid

## 📧 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Make sure you're using a modern browser (Chrome/Edge recommended)
4. Check that your microphone is working in other applications

## 🔄 Version Information

- **Streamlit**: 1.28.0 or higher required
- **Python**: 3.8 or higher recommended
- **Browser**: Chrome, Edge, or Firefox (latest versions)

---

**Good luck with your percussion testing experiment! 🎯**
