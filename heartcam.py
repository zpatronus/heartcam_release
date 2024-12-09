import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import time


# filter out freq<1Hz or >3Hz
def bandpass_filter(data, lowcut, highcut, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


# get freq - magnitude
def get_fft(signal, sampling_rate):
    fft_result = fft([x[0] for x in signal])
    freqs = fftfreq(len(signal), d=1 / sampling_rate)
    positive_freqs = freqs[freqs >= 0]
    magnitudes = np.abs(fft_result[freqs >= 0])
    return positive_freqs, magnitudes


def extract_red_dominance(frame):
    red_channel = frame[:, :, 2].astype(float)
    green_channel = frame[:, :, 1].astype(float)
    blue_channel = frame[:, :, 0].astype(float)
    red_dominance = (red_channel > green_channel) & (red_channel > blue_channel)
    red_dominance_percentage = (np.sum(red_dominance) / red_channel.size) * 100
    return red_dominance_percentage


def extract_rgb_average(frame):
    red_avg = np.mean(frame[:, :, 2])
    green_avg = np.mean(frame[:, :, 1])
    blue_avg = np.mean(frame[:, :, 0])
    return (red_avg + green_avg + blue_avg) / 3


def smooth_heart_rate(current_rate, new_rate, alpha=0.05):
    return alpha * new_rate + (1 - alpha) * current_rate


# camera selection interface with detailed camera info
def select_camera():
    index = 0
    available_cameras = []

    print("Checking for available cameras...")
    camera_details = {}
    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            camera_details[index] = (width, height, fps)

        cap.release()
        index += 1

    if not available_cameras:
        print("No cameras found.")
        return None

    print("\nAvailable cameras:")
    for i in available_cameras:
        width, height, fps = camera_details[i]
        print(f"Camera {i}: Resolution: {int(width)}x{int(height)}, FPS: {fps:.2f}")

    camera_id = int(input("Select the camera ID: "))
    while camera_id not in available_cameras:
        print("Invalid camera ID. Please select a valid camera ID.")
        camera_id = int(input("Select the camera ID: "))

    return camera_id


camera_id = select_camera()
if camera_id is None:
    print("No available camera to select. Exiting.")
    exit()

cap = cv2.VideoCapture(camera_id)

fps = cap.get(cv2.CAP_PROP_FPS)

signal = []
# larger buffer -> longer sampling time & more sensitive to motion, but smaller steps
sampling_time = 10
abandon_time = 0
abandon_per_frame = 2
buffer_size = 256
freq_min, freq_max = 0.66, 3
smoothed_heart_rate = None
finger_detected = False
countdown_start_time = None

plt.ion()
fig, ax = plt.subplots()
(line,) = ax.plot([], [])
smoothed_hr_line = ax.axvline(x=0, color="r", linestyle="--", label="Smoothed HR")
smoothed_hr_text = ax.text(0, 0.5, "", color="r", fontsize=10, va="center")

ax.set_ylim(0, 1)
ax.set_xlim(freq_min * 60, freq_max * 60)
ax.set_xlabel("Heart Rate (BPM)")
ax.set_ylabel("Normalized Magnitude")
plt.title("Live Heart Rate Probability (2/3 - 3 Hz, 40 - 180 BPM)")
ax.grid(True)
ax.set_xticks(np.arange(40, 181, 10))

pop_signal = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    red_dominance_percentage = extract_red_dominance(frame)
    if red_dominance_percentage >= 99.5:
        # there's finger over the camera
        if not finger_detected:
            # just put on
            finger_detected = True
            countdown_start_time = time.time()
            signal.clear()

        # obtain frame from camera
        rgb_value = extract_rgb_average(frame)
        signal.append((rgb_value, time.time()))
        if len(signal) > buffer_size:
            signal.pop(0)
        sampling_rate = len(signal) / (
            1
            if int(signal[-1][1] - signal[0][1]) == 0
            else int(signal[-1][1] - signal[0][1])
        )
        print("Sampling rate: ", sampling_rate, " Buffer length: ", len(signal))

        elapsed_time = time.time() - countdown_start_time

        # f"Hold still and cover the camera... {int(sampling_time - elapsed_time)}s left",
        if elapsed_time < sampling_time:
            # wait, hold the finger
            cv2.putText(
                frame,
                "Hold still and cover the camera...",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            if elapsed_time > abandon_time:
                if pop_signal == abandon_per_frame:
                    signal.pop(0)
                    pop_signal = 0
                else:
                    pop_signal += 1
        if elapsed_time > max(abandon_time, 1):
            # output heart rate
            positive_freqs, magnitudes = get_fft(signal, sampling_rate=sampling_rate)

            indices = (positive_freqs >= freq_min) & (positive_freqs <= freq_max)
            filtered_freqs = positive_freqs[indices]
            filtered_magnitudes = magnitudes[indices]
            max_magnitude = np.max(filtered_magnitudes)
            if max_magnitude != 0:
                filtered_magnitudes = filtered_magnitudes / max_magnitude
            peak_freq = filtered_freqs[np.argmax(filtered_magnitudes)]
            heart_rate_bpm = peak_freq * 60

            if smoothed_heart_rate is None:
                smoothed_heart_rate = heart_rate_bpm
            else:
                smoothed_heart_rate = smooth_heart_rate(
                    smoothed_heart_rate, heart_rate_bpm
                )

            line.set_data(filtered_freqs * 60, filtered_magnitudes)

            smoothed_hr_line.set_data(
                [smoothed_heart_rate, smoothed_heart_rate], [0, 1]
            )
            smoothed_hr_text.set_position((smoothed_heart_rate, 0.5))
            smoothed_hr_text.set_text(f"{smoothed_heart_rate:.2f} BPM")

            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001)

            cv2.putText(
                frame,
                f"Heart Rate: {smoothed_heart_rate:.2f} bpm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
    else:
        # there's no finger over the camera
        finger_detected = False
        countdown_start_time = None
        pop_signal = 0
        cv2.putText(
            frame,
            "Please place your finger on the camera",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
