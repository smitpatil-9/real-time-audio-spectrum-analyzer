# Real-Time Audio Spectrum Analyzer

A Python-based real-time audio signal processing system that captures microphone input and analyzes it in both the time and frequency domains.

## Features

- Real-time microphone signal acquisition
- Time-domain waveform visualization
- FFT-based frequency-domain analysis
- Dominant frequency detection
- Fundamental pitch estimation using autocorrelation
- RMS signal-level measurement
- Frequency-band energy distribution
- Butterworth band-pass digital filtering
- Original vs filtered spectrum comparison
- Real-time spectrogram
- Live 2×2 analysis dashboard

## System Overview

```text
Microphone
    ↓
Audio Sampling
    ↓
Digital Audio Buffer
    ↓
 ┌───────────────┬───────────────┬───────────────┐
 ↓               ↓               ↓
Time Domain     FFT             RMS
 ↓               ↓
Waveform       Spectrum
                 ↓
       ┌─────────┴─────────┐
       ↓                   ↓
Dominant Frequency    Band Energy
       ↓
Pitch Detection
       
FFT Signal
    ↓
Butterworth Band-Pass Filter
    ↓
Filtered Spectrum

Audio Signal
    ↓
Time-Frequency Analysis
    ↓
Spectrogram