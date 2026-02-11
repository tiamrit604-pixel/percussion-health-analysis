# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### 🎤 Microphone Issues

#### Problem: "Microphone not detected" or no recording button appears

**Solutions:**
1. **Check Browser Permissions**
   - Chrome: Click the lock icon in the address bar → Site settings → Microphone → Allow
   - Firefox: Click the lock icon → Permissions → Use the Microphone → Allow
   - Edge: Click the lock icon → Permissions for this site → Microphone → Allow

2. **Check System Microphone**
   - Windows: Settings → Privacy → Microphone → Allow apps to access microphone
   - Mac: System Preferences → Security & Privacy → Privacy → Microphone
   - Linux: Check PulseAudio/ALSA settings

3. **Try a Different Browser**
   - Chrome and Edge have the best support for `st.audio_input()`
   - Update to the latest browser version

4. **Restart the App**
   ```bash
   # Stop the app (Ctrl+C)
   # Run again
   streamlit run percussion_browser_recording.py
   ```

#### Problem: Recording is silent or very quiet

**Solutions:**
1. Check microphone is not muted in system settings
2. Increase microphone gain/volume in system settings
3. Move microphone closer to the plate (10-15 cm optimal)
4. Try an external microphone if built-in mic is poor quality

---

### 💻 Installation Issues

#### Problem: `pip install` fails

**Solutions:**
1. **Update pip first**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install one package at a time** to identify the problem
   ```bash
   pip install streamlit
   pip install librosa
   pip install matplotlib
   # etc.
   ```

3. **Use conda instead** (if you have Anaconda/Miniconda)
   ```bash
   conda create -n percussion python=3.9
   conda activate percussion
   conda install -c conda-forge streamlit librosa matplotlib scipy pywavelets
   ```

#### Problem: "ModuleNotFoundError" when running

**Solutions:**
1. Make sure virtual environment is activated
   ```bash
   # Linux/Mac
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

2. Reinstall requirements
   ```bash
   pip install -r requirements.txt
   ```

3. Check Python version (needs 3.8+)
   ```bash
   python --version
   ```

---

### 🌐 Browser Issues

#### Problem: App doesn't open in browser

**Solutions:**
1. Manually open browser and navigate to: `http://localhost:8501`
2. Check if port 8501 is already in use:
   ```bash
   # Use a different port
   streamlit run percussion_browser_recording.py --server.port 8502
   ```

#### Problem: "Connection refused" or can't reach localhost

**Solutions:**
1. Check firewall settings
2. Try: `http://127.0.0.1:8501` instead of `localhost`
3. Restart the Streamlit server

---

### 📊 Analysis Issues

#### Problem: "Signal too short after processing"

**Solutions:**
1. Record for longer (at least 2 seconds)
2. Hit the plate harder to ensure a clear onset
3. Reduce background noise
4. Check that you actually hit the plate during recording

#### Problem: "No clear onset detected"

**Solutions:**
1. Hit the plate more firmly
2. Reduce time between starting recording and hitting
3. Move microphone closer to impact point
4. Try manual analysis by selecting a different onset detection threshold

#### Problem: Plots look strange or empty

**Solutions:**
1. Check audio quality by listening to the recording first
2. Ensure recording isn't clipped (too loud)
3. Try recording again with better technique
4. Verify sample rate is reasonable (should be 44100 or 48000 Hz)

---

### 💾 Data Issues

#### Problem: Can't download recordings

**Solutions:**
1. Check browser download settings
2. Try a different browser
3. Look in your default downloads folder
4. Disable any download-blocking browser extensions

#### Problem: Recordings are lost after closing browser

**This is expected behavior!** The app stores recordings in session state which is temporary.

**Solutions:**
1. Download recordings immediately after collection
2. Don't close/refresh the browser tab until you've downloaded everything
3. Consider implementing persistent storage (database) for production use

---

### 🎯 Performance Issues

#### Problem: App is slow or freezing

**Solutions:**
1. **Close unused tabs** - Streamlit can be memory-intensive
2. **Analyze one recording at a time**
3. **Clear old recordings** using the "Clear All Recordings" button
4. **Reduce spectrogram resolution** in the code:
   ```python
   # Change this line in analyze_audio function
   D = librosa.stft(y_processed, n_fft=1024, hop_length=256)  # Lower values
   ```

#### Problem: Out of memory errors

**Solutions:**
1. Restart the app
2. Record shorter clips (0.5-2 seconds is sufficient)
3. Close other memory-intensive applications
4. Process recordings one at a time instead of batch processing

---

### 🐛 Specific Error Messages

#### Error: `AttributeError: 'NoneType' object has no attribute 'read'`

**Cause:** No audio was recorded
**Solution:** Make sure to actually record audio before clicking analyze

#### Error: `InvalidIndexError` or `IndexError`

**Cause:** Trying to analyze when no recordings exist
**Solution:** Record at least one percussion sound first

#### Error: `StreamlitAPIException: Audio input not supported`

**Cause:** Streamlit version is too old
**Solution:** 
```bash
pip install --upgrade streamlit
# Requires Streamlit >= 1.28.0
```

#### Error: `librosa.load` fails

**Cause:** Audio format not supported or corrupt recording
**Solutions:**
1. Update librosa: `pip install --upgrade librosa soundfile`
2. Try recording again
3. Check that microphone is working in other apps

---

### 📱 Browser Compatibility

**Recommended Browsers (in order):**
1. ✅ Google Chrome (Latest)
2. ✅ Microsoft Edge (Latest)
3. ⚠️ Firefox (Latest) - audio recording may be less stable
4. ❌ Safari - `st.audio_input()` may not work

**Not Supported:**
- Internet Explorer
- Very old browser versions
- Mobile browsers (may work but not optimized)

---

### 🆘 Still Having Issues?

1. **Check Streamlit version:**
   ```bash
   streamlit --version
   # Should be >= 1.28.0
   ```

2. **Check all dependencies:**
   ```bash
   pip list
   ```

3. **Try the example script:**
   ```bash
   streamlit hello
   # If this works, issue is with your code
   # If this doesn't work, issue is with Streamlit installation
   ```

4. **Clear Streamlit cache:**
   ```bash
   streamlit cache clear
   ```

5. **Run with debug mode:**
   ```bash
   streamlit run percussion_browser_recording.py --logger.level=debug
   ```

6. **Create a new virtual environment:**
   ```bash
   # Delete old venv
   rm -rf venv  # Linux/Mac
   rmdir /s venv  # Windows
   
   # Create fresh environment
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

---

### 📧 Getting Help

If none of these solutions work:

1. Note the exact error message
2. Note your Python version: `python --version`
3. Note your OS (Windows, Mac, Linux)
4. Note your browser and version
5. Check Streamlit community forum: https://discuss.streamlit.io/

---

## Quick Reference Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run percussion_browser_recording.py

# Run on different port
streamlit run percussion_browser_recording.py --server.port 8502

# Update Streamlit
pip install --upgrade streamlit

# Check versions
python --version
streamlit --version
pip list

# Clear cache
streamlit cache clear
```
