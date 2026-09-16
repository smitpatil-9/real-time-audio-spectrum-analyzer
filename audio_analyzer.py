import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, spectrogram

FS = 16000
DURATION = 5
DEVICE = 1

print("================================")
print("   AUDIO SIGNAL ANALYZER")
print("================================")

input("\nPress ENTER to start recording...")

print("\n🎤 Recording... SPEAK NOW!")

audio = sd.rec(
    int(DURATION * FS),
    samplerate=FS,
    channels=1,
    dtype="float32",
    device=DEVICE
)

sd.wait()

print("✅ Recording completed!")

audio = audio.flatten()

# Number of samples
N = len(audio)

# Time axis
t = np.arange(N) / FS

# =====================================================
# TIME DOMAIN
# =====================================================

plt.figure(figsize=(12, 5))
plt.plot(t, audio)

plt.title("Audio Signal - Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

# =====================================================
# FFT - FREQUENCY DOMAIN
# =====================================================

fft_result = np.fft.rfft(audio)

magnitude = np.abs(fft_result)

frequencies = np.fft.rfftfreq(N, 1 / FS)

# Ignore DC component
magnitude[0] = 0

# Find dominant frequency
dominant_index = np.argmax(magnitude)

dominant_frequency = frequencies[dominant_index]

print("\n================================")
print("      FREQUENCY ANALYSIS")
print("================================")

print(f"Dominant Frequency : {dominant_frequency:.2f} Hz")

# =====================================================
# RMS SIGNAL LEVEL
# =====================================================

rms = np.sqrt(np.mean(audio**2))

print(f"RMS Signal Level  : {rms:.4f}")

# =====================================================
# FREQUENCY BAND ANALYSIS
# =====================================================

bands = [
    (0, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000)
]

print("\n================================")
print("     FREQUENCY BAND ANALYSIS")
print("================================")

band_energy = []

for low, high in bands:

    # Select FFT frequencies inside the band
    mask = (frequencies >= low) & (frequencies < high)

    # Calculate energy
    energy = np.sum(magnitude[mask] ** 2)

    band_energy.append(energy)

    print(f"{low:4d} - {high:4d} Hz : {energy:.2f}")

# =====================================================
# FREQUENCY BAND ENERGY PLOT
# =====================================================

band_labels = [
    "0-250 Hz",
    "250-500 Hz",
    "500-1000 Hz",
    "1-2 kHz",
    "2-4 kHz",
    "4-8 kHz"
]

plt.figure(figsize=(10, 5))

plt.bar(band_labels, band_energy)

plt.title("Energy Distribution Across Frequency Bands")
plt.xlabel("Frequency Band")
plt.ylabel("Spectral Energy")

plt.xticks(rotation=20)
plt.grid(axis="y")

plt.show()

# =====================================================
# DIGITAL BAND-PASS FILTER
# =====================================================

from scipy.signal import butter, filtfilt

LOW_CUTOFF = 100
HIGH_CUTOFF = 4000
FILTER_ORDER = 6

# Normalize cutoff frequencies
nyquist = FS / 2

low = LOW_CUTOFF / nyquist
high = HIGH_CUTOFF / nyquist

# Design Butterworth filter
b, a = butter(
    FILTER_ORDER,
    [low, high],
    btype="bandpass"
)

# Apply filter
filtered_audio = filtfilt(b, a, audio)

print("\n================================")
print("       DIGITAL FILTERING")
print("================================")

print(f"Filter Type : Butterworth Band-Pass")
print(f"Filter Order : {FILTER_ORDER}")
print(f"Passband : {LOW_CUTOFF} - {HIGH_CUTOFF} Hz")

# =====================================================
# ORIGINAL vs FILTERED - TIME DOMAIN
# =====================================================

plt.figure(figsize=(12, 5))

plt.plot(t, audio, label="Original")
plt.plot(t, filtered_audio, label="Filtered")

plt.title("Original vs Filtered Audio - Time Domain")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()
plt.grid()

plt.show()

# =====================================================
# FFT COMPARISON - BEFORE vs AFTER FILTERING
# =====================================================

# FFT of original signal
fft_original = np.fft.rfft(audio)
magnitude_original = np.abs(fft_original)

# FFT of filtered signal
fft_filtered = np.fft.rfft(filtered_audio)
magnitude_filtered = np.abs(fft_filtered)

# Frequency axis
freq_original = np.fft.rfftfreq(len(audio), 1 / FS)
freq_filtered = np.fft.rfftfreq(len(filtered_audio), 1 / FS)

# =====================================================
# FILTER FREQUENCY RESPONSE
# =====================================================

from scipy.signal import freqz

w, h = freqz(b, a, worN=2000)

filter_frequencies = w * FS / (2 * np.pi)

plt.figure(figsize=(12, 5))

plt.plot(filter_frequencies, 20 * np.log10(np.maximum(np.abs(h), 1e-10)))

plt.title("Butterworth Band-Pass Filter Frequency Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

plt.xlim(0, 8000)
plt.ylim(-80, 5)

plt.axvline(100, linestyle="--", label="100 Hz cutoff")
plt.axvline(4000, linestyle="--", label="4000 Hz cutoff")

plt.legend()
plt.grid()

plt.show()

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(12, 5))

plt.plot(
    freq_original,
    magnitude_original,
    label="Original"
)

plt.plot(
    freq_filtered,
    magnitude_filtered,
    label="Filtered"
)

plt.title("Original vs Filtered Audio - Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.xlim(0, 8000)

plt.legend()
plt.grid()

plt.show()

# =====================================================
# FREQUENCY DOMAIN PLOT
# =====================================================

plt.figure(figsize=(12, 5))

plt.plot(frequencies, magnitude)

plt.title("Audio Signal - Frequency Domain (FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.xlim(0, 4000)

plt.grid()

plt.show()

# =====================================================
# SPECTROGRAM
# =====================================================

from scipy.signal import spectrogram

f, tt, Sxx = spectrogram(
    audio,
    fs=FS,
    nperseg=256,
    noverlap=200
)

plt.figure(figsize=(12, 6))

plt.pcolormesh(
    tt,
    f,
    10 * np.log10(Sxx + 1e-10),
    shading="gouraud"
)

plt.title("Audio Signal Spectrogram")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")

plt.ylim(0, 4000)

plt.colorbar(label="Power (dB)")

plt.show()

# =====================================================
# PITCH DETECTION USING AUTOCORRELATION
# =====================================================

# Take the middle portion of the recording
start = int(1.0 * FS)
end = int(2.0 * FS)

pitch_signal = audio[start:end]

# Remove DC component
pitch_signal = pitch_signal - np.mean(pitch_signal)

# Autocorrelation
correlation = np.correlate(
    pitch_signal,
    pitch_signal,
    mode="full"
)

# Keep positive lags
correlation = correlation[len(correlation) // 2:]

# Expected human voice range
MIN_PITCH = 80
MAX_PITCH = 400

min_lag = int(FS / MAX_PITCH)
max_lag = int(FS / MIN_PITCH)

# Find strongest repeating period
peak_lag = min_lag + np.argmax(
    correlation[min_lag:max_lag]
)

# Calculate fundamental frequency
pitch = FS / peak_lag

print("\n================================")
print("        PITCH DETECTION")
print("================================")

print(f"Estimated Fundamental Frequency : {pitch:.2f} Hz")

# Plot autocorrelation
lags = np.arange(len(correlation))

plt.figure(figsize=(12, 5))

plt.plot(lags / FS, correlation)

plt.xlim(0, 0.025)

plt.title("Autocorrelation for Pitch Detection")
plt.xlabel("Time Lag (seconds)")
plt.ylabel("Correlation")

plt.grid()

plt.show()

