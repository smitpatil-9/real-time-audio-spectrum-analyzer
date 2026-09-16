import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt


# ============================================================
# REAL-TIME AUDIO SPECTRUM ANALYZER
# ============================================================

FS = 16000
BLOCK_SIZE = 1024
BUFFER_SIZE = 4096
DEVICE = 1

# Silence threshold
RMS_THRESHOLD = 0.003

# Pitch range for voiced human speech
MIN_PITCH = 80
MAX_PITCH = 400


# ============================================================
# AUDIO BUFFER
# ============================================================

audio_buffer = np.zeros(BUFFER_SIZE)


# ============================================================
# PITCH SMOOTHING
# ============================================================

smoothed_pitch = None


# ============================================================
# BUTTERWORTH BAND-PASS FILTER
# ============================================================

LOW_CUTOFF = 100
HIGH_CUTOFF = 4000
FILTER_ORDER = 6

nyquist = FS / 2

low = LOW_CUTOFF / nyquist
high = HIGH_CUTOFF / nyquist

b, a = butter(
    FILTER_ORDER,
    [low, high],
    btype="bandpass"
)


# ============================================================
# FREQUENCY BANDS
# ============================================================

bands = [
    (0, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000)
]

band_labels = [
    "0-250",
    "250-500",
    "500-1k",
    "1-2k",
    "2-4k"
]


# ============================================================
# AUDIO CALLBACK
# ============================================================

def audio_callback(indata, frames, time, status):

    global audio_buffer

    if status:
        print(status)

    new_audio = indata[:, 0]

    audio_buffer = np.roll(
        audio_buffer,
        -frames
    )

    audio_buffer[-frames:] = new_audio


# ============================================================
# TIME / FREQUENCY AXES
# ============================================================

time_axis = np.arange(
    BUFFER_SIZE
) / FS

freq_axis = np.fft.rfftfreq(
    BUFFER_SIZE,
    1 / FS
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 9)
)

ax_time = axes[0, 0]
ax_fft = axes[0, 1]
ax_band = axes[1, 0]
ax_spec = axes[1, 1]


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "REAL-TIME AUDIO SPECTRUM ANALYZER",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# TIME DOMAIN
# ============================================================

time_line, = ax_time.plot(
    time_axis,
    audio_buffer
)

ax_time.set_title(
    "Time-Domain Waveform"
)

ax_time.set_xlabel(
    "Time (seconds)"
)

ax_time.set_ylabel(
    "Amplitude"
)

ax_time.set_ylim(
    -0.1,
    0.1
)

ax_time.grid()


# ============================================================
# FFT
# ============================================================

fft_line, = ax_fft.plot(
    freq_axis,
    np.zeros(len(freq_axis)),
    label="Original"
)

filtered_fft_line, = ax_fft.plot(
    freq_axis,
    np.zeros(len(freq_axis)),
    label="Filtered"
)

ax_fft.set_title(
    "Frequency-Domain Spectrum"
)

ax_fft.set_xlabel(
    "Frequency (Hz)"
)

ax_fft.set_ylabel(
    "Magnitude"
)

ax_fft.set_xlim(
    0,
    4000
)

ax_fft.set_ylim(
    0,
    10
)

ax_fft.legend(
    loc="upper right"
)

ax_fft.grid()


# ============================================================
# FILTER INDICATOR
# ============================================================

filter_text = ax_fft.text(
    0.02,
    0.94,
    "Butterworth Band-Pass: 100–4000 Hz",
    transform=ax_fft.transAxes,
    fontsize=9,
    verticalalignment="top"
)


# ============================================================
# FREQUENCY BAND ENERGY
# ============================================================

band_bars = ax_band.bar(
    band_labels,
    np.zeros(len(bands))
)

ax_band.set_title(
    "Frequency Band Energy Distribution"
)

ax_band.set_xlabel(
    "Frequency Band"
)

ax_band.set_ylabel(
    "Energy (% of total)"
)

ax_band.set_ylim(
    0,
    100
)

ax_band.grid(
    axis="y"
)


# ============================================================
# SPECTROGRAM
# ============================================================

SPEC_ROWS = 100
SPEC_COLS = 100

spectrogram_data = np.full(
    (SPEC_ROWS, SPEC_COLS),
    -100.0
)

spec_image = ax_spec.imshow(
    spectrogram_data,
    origin="lower",
    aspect="auto",
    extent=[
        0,
        1,
        0,
        4000
    ],
    vmin=-100,
    vmax=0
)

ax_spec.set_title(
    "Real-Time Spectrogram"
)

ax_spec.set_xlabel(
    "Time (relative)"
)

ax_spec.set_ylabel(
    "Frequency (Hz)"
)


# ============================================================
# COLORBAR
# ============================================================

fig.colorbar(
    spec_image,
    ax=ax_spec,
    label="Power (dB)"
)


# ============================================================
# PITCH DETECTION FUNCTION
# ============================================================

def detect_pitch(signal):

    signal = signal - np.mean(signal)

    energy = np.sum(
        signal ** 2
    )

    if energy < 1e-8:
        return 0, 0

    # Autocorrelation
    correlation = np.correlate(
        signal,
        signal,
        mode="full"
    )

    correlation = correlation[
        len(correlation) // 2:
    ]

    # Normalize autocorrelation
    correlation = correlation / (
        correlation[0] + 1e-12
    )

    min_lag = int(
        FS / MAX_PITCH
    )

    max_lag = int(
        FS / MIN_PITCH
    )

    if max_lag >= len(correlation):
        max_lag = len(correlation) - 1

    search_region = correlation[
        min_lag:max_lag
    ]

    if len(search_region) == 0:
        return 0, 0

    peak_index = np.argmax(
        search_region
    )

    pitch_lag = (
        min_lag
        + peak_index
    )

    confidence = correlation[
        pitch_lag
    ]

    pitch = FS / pitch_lag

    return pitch, confidence


# ============================================================
# UPDATE FUNCTION
# ============================================================

def update(frame):

    global audio_buffer
    global smoothed_pitch
    global spectrogram_data

    # ========================================================
    # TIME DOMAIN
    # ========================================================

    time_line.set_ydata(
        audio_buffer
    )


    # ========================================================
    # RMS
    # ========================================================

    rms = np.sqrt(
        np.mean(
            audio_buffer ** 2
        )
    )


    # ========================================================
    # HANN WINDOW FOR FFT
    # ========================================================

    window = np.hanning(
        BUFFER_SIZE
    )

    windowed_audio = (
        audio_buffer * window
    )


    # ========================================================
    # ORIGINAL FFT
    # ========================================================

    fft_result = np.fft.rfft(
        windowed_audio
    )

    original_magnitude = np.abs(
        fft_result
    )


    # ========================================================
    # DOMINANT FREQUENCY
    # Only analyze the displayed range: 0–4000 Hz
    # ========================================================

    dominant_search = original_magnitude.copy()

    # Ignore DC component
    dominant_search[0] = 0

    # Ignore frequencies above 4000 Hz
    dominant_search[freq_axis > HIGH_CUTOFF] = 0

    dominant_index = np.argmax(
        dominant_search
    )

    dominant_frequency = (
        freq_axis[dominant_index]
    )


    # ========================================================
    # DIGITAL FILTER
    # ========================================================

    try:

        filtered_audio = filtfilt(
            b,
            a,
            audio_buffer
        )

    except ValueError:

        filtered_audio = (
            audio_buffer.copy()
        )


    # ========================================================
    # FILTERED FFT
    # ========================================================

    filtered_windowed = (
        filtered_audio * window
    )

    filtered_fft = np.fft.rfft(
        filtered_windowed
    )

    filtered_magnitude = np.abs(
        filtered_fft
    )


    # ========================================================
    # UPDATE FFT
    # ========================================================

    fft_line.set_ydata(
        original_magnitude
    )

    filtered_fft_line.set_ydata(
        filtered_magnitude
    )


    # ========================================================
    # FFT SCALE
    # ========================================================

    max_magnitude = max(
        np.max(original_magnitude),
        np.max(filtered_magnitude)
    )

    if max_magnitude > 0:

        ax_fft.set_ylim(
            0,
            max_magnitude * 1.2
        )


    # ========================================================
    # PITCH DETECTION
    # ========================================================

    pitch = 0
    pitch_confidence = 0

    if rms > RMS_THRESHOLD:

        pitch, pitch_confidence = detect_pitch(
            audio_buffer
        )

        # Require reasonable autocorrelation confidence
        if pitch_confidence < 0.30:

            pitch = 0


    # ========================================================
    # PITCH SMOOTHING
    # ========================================================

    if pitch > 0:

        if smoothed_pitch is None:

            smoothed_pitch = pitch

        else:

            smoothed_pitch = (
                0.75 * smoothed_pitch
                + 0.25 * pitch
            )

    else:

        smoothed_pitch = None


    # ========================================================
    # SILENCE / VOICE STATUS
    # ========================================================

    if rms <= RMS_THRESHOLD:

        signal_status = "STATUS: SILENCE / NO VOICE"

    elif smoothed_pitch is not None:

        signal_status = "STATUS: VOICE DETECTED"

    else:

        signal_status = "STATUS: UNVOICED / NO STABLE PITCH"


    # ========================================================
    # MAIN DISPLAY STATUS
    # ========================================================

    if smoothed_pitch is not None:

        status_text = (
            f"Dominant: {dominant_frequency:.1f} Hz   |   "
            f"Pitch: {smoothed_pitch:.1f} Hz   |   "
            f"RMS: {rms:.4f}\n"
            f"{signal_status}"
        )

    else:

        status_text = (
            f"Dominant: {dominant_frequency:.1f} Hz   |   "
            f"Pitch: --   |   "
            f"RMS: {rms:.4f}\n"
            f"{signal_status}"
        )


    fig.suptitle(
        "REAL-TIME AUDIO SPECTRUM ANALYZER\n"
        + status_text,
        fontsize=15,
        fontweight="bold"
    )


    # ========================================================
    # FREQUENCY BAND ENERGY
    # ========================================================

    band_energy = []

    for low_band, high_band in bands:

        mask = (
            (freq_axis >= low_band)
            &
            (freq_axis < high_band)
        )

        energy = np.sum(
            original_magnitude[mask] ** 2
        )

        band_energy.append(
            energy
        )


    # ========================================================
    # NORMALIZE TO PERCENTAGE
    # ========================================================

    total_energy = sum(
        band_energy
    )

    if total_energy > 0:

        band_percent = [
            (energy / total_energy) * 100
            for energy in band_energy
        ]

    else:

        band_percent = [
            0
            for _ in band_energy
        ]


    # ========================================================
    # UPDATE BAND BARS
    # ========================================================

    for bar, percentage in zip(
        band_bars,
        band_percent
    ):

        bar.set_height(
            percentage
        )


    # ========================================================
    # REAL-TIME SPECTROGRAM
    # ========================================================

    spectrum = np.abs(
        np.fft.rfft(
            windowed_audio
        )
    )

    power_db = (
        20
        * np.log10(
            spectrum + 1e-6
        )
    )


    # Only display frequencies up to 4000 Hz
    frequency_mask = (
        freq_axis <= 4000
    )

    current_spectrum = (
        power_db[frequency_mask]
    )


    # Resize to spectrogram rows
    resized_spectrum = np.interp(
        np.linspace(
            0,
            len(current_spectrum) - 1,
            SPEC_ROWS
        ),
        np.arange(
            len(current_spectrum)
        ),
        current_spectrum
    )


    # Scroll left
    spectrogram_data = np.roll(
        spectrogram_data,
        -1,
        axis=1
    )


    # Add latest spectrum
    spectrogram_data[:, -1] = (
        resized_spectrum
    )


    # Update image
    spec_image.set_data(
        spectrogram_data
    )


    return (
        time_line,
        fft_line,
        filtered_fft_line,
        *band_bars,
        spec_image
    )


# ============================================================
# START MESSAGE
# ============================================================

print()
print("================================================")
print("       REAL-TIME AUDIO SPECTRUM ANALYZER")
print("================================================")
print()
print("Microphone Device :", DEVICE)
print("Sampling Frequency:", FS, "Hz")
print("Buffer Size       :", BUFFER_SIZE)
print()
print("SIGNAL PROCESSING:")
print("  ✓ Time-domain analysis")
print("  ✓ FFT frequency-domain analysis")
print("  ✓ Dominant frequency detection")
print("  ✓ RMS signal-level measurement")
print("  ✓ Fundamental pitch estimation")
print("  ✓ Frequency-band energy analysis")
print("  ✓ Butterworth band-pass filtering")
print("  ✓ Real-time spectrogram")
print()
print("FILTER:")
print(f"  Butterworth order : {FILTER_ORDER}")
print(f"  Passband          : {LOW_CUTOFF}–{HIGH_CUTOFF} Hz")
print()
print("Speak into the microphone.")
print("Close the graph window to stop.")
print()


# ============================================================
# MICROPHONE STREAM
# ============================================================

stream = sd.InputStream(
    samplerate=FS,
    blocksize=BLOCK_SIZE,
    channels=1,
    dtype="float32",
    device=DEVICE,
    callback=audio_callback
)

stream.start()


# ============================================================
# ANIMATION
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    interval=50,
    blit=False,
    cache_frame_data=False
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.91]
)


# ============================================================
# DISPLAY + SAFE SHUTDOWN
# ============================================================

try:

    plt.show()

finally:

    stream.stop()
    stream.close()

    print()
    print("Audio analyzer stopped.")