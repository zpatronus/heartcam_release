import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import time
import datetime
import json
import argparse

parser = argparse.ArgumentParser(
    description="Measure heart rate by forehead detection."
)

parser.add_argument(
    "output_file",
    default="heart_rate_log.json",
    type=str,
    nargs="?",
    help="Path to the output JSON file.",
)

output_file = parser.parse_args().output_file


def detect_forehead(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    for x, y, w, h in faces:
        forehead_x1 = x
        forehead_y1 = y
        forehead_x2 = x + w
        forehead_y2 = y + int(0.3 * h)

        region_width = w // 5
        region_height = h // 10
        small_forehead_x1 = forehead_x1 + (w - region_width) // 2
        small_forehead_y1 = forehead_y1 + (int(0.3 * h) - region_height) // 2
        small_forehead_x2 = small_forehead_x1 + region_width
        small_forehead_y2 = small_forehead_y1 + region_height

        small_forehead_region = frame[
            small_forehead_y1:small_forehead_y2, small_forehead_x1:small_forehead_x2
        ]
        return small_forehead_region, (
            small_forehead_x1,
            small_forehead_y1,
            small_forehead_x2,
            small_forehead_y2,
        )
    return None, None


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def bandpass_filter(data, lowcut, highcut, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def get_fft(signal, sampling_rate):
    fft_result = fft([x[0] for x in signal])
    freqs = fftfreq(len(signal), d=1 / sampling_rate)
    positive_freqs = freqs[freqs >= 0]
    magnitudes = np.abs(fft_result[freqs >= 0])
    return positive_freqs, magnitudes


def find_peaks(min_freq, max_freq, filtered_freqs, filtered_magnitudes):
    peak_index = None
    for i in range(len(filtered_freqs)):
        if filtered_freqs[i] < min_freq:
            continue
        if filtered_freqs[i] > max_freq:
            break
        if filtered_magnitudes[i] < 0.75:
            continue
        if peak_index is None:
            peak_index = i
        else:
            if filtered_magnitudes[i] > filtered_magnitudes[peak_index]:
                peak_index = i
    return peak_index


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
        print(
            f"Camera {i}: Resolution: {int(width)}x{int(height)}, FPS: {fps:.2f}"
        )

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
sampling_time = 10
abandon_time = 0
abandon_per_frame = 2
buffer_size = 256
freq_min, freq_max = 0.66, 3
smoothed_heart_rate = None
finger_detected = False
countdown_start_time = None
smoothedBrightness = 0
last_pulse_time = 0

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

result_log = []


def sliding_window(sampling_rate, signal, bpm):
    tics = 1 + int(min((1 / (bpm / 60)) * sampling_rate / 3, buffer_size))
    return np.mean(signal[-tics:])


def exp_smooth(currentRate, newRate, samplingRate, alpha=1):
    return (
        0.52 / samplingRate * alpha * newRate
        + (1 - 0.52 / samplingRate * alpha) * currentRate
    )


def camera_stop(event):
    global running
    running = False


ax_stop = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_stop = Button(ax_stop, "Stop")
btn_stop.on_clicked(camera_stop)

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break
    small_forehead_region, small_forehead_coords = detect_forehead(frame, face_cascade)
    if small_forehead_region is not None:

        red_dominance_percentage = extract_red_dominance(small_forehead_region)
        cv2.rectangle(
            frame,
            (small_forehead_coords[0], small_forehead_coords[1]),
            (small_forehead_coords[2], small_forehead_coords[3]),
            (255, 0, 0),
            2,
        )
        if red_dominance_percentage >= 99.5:
            if not finger_detected:
                finger_detected = True
                countdown_start_time = time.time()
                signal.clear()

            rgb_value = extract_rgb_average(small_forehead_region)
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

            if elapsed_time < sampling_time:
                cv2.putText(
                    frame,
                    "Hold still and cover the camera...",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    4,
                )
                if elapsed_time > abandon_time:
                    if pop_signal == abandon_per_frame:
                        signal.pop(0)
                        pop_signal = 0
                    else:
                        pop_signal += 1
            if elapsed_time > max(abandon_time, 1):
                positive_freqs, magnitudes = get_fft(
                    signal, sampling_rate=sampling_rate
                )

                indices = (positive_freqs >= freq_min) & (positive_freqs <= freq_max)
                filtered_freqs = positive_freqs[indices]
                filtered_magnitudes = magnitudes[indices]
                max_magnitude = np.max(filtered_magnitudes)
                temp_peak_index = np.argmax(filtered_magnitudes)
                peak_index = None
                if filtered_freqs[temp_peak_index] > 120.0 / 60:
                    peak_index = find_peaks(
                        filtered_freqs[temp_peak_index] / 2.0 - 10.0 / 60,
                        filtered_freqs[temp_peak_index] / 2.0 + 10.0 / 60,
                        filtered_freqs,
                        filtered_magnitudes,
                    )
                if filtered_freqs[temp_peak_index] < 50.0 / 60:
                    peak_index = find_peaks(
                        60.0 / 60, 100.0 / 60, filtered_freqs, filtered_magnitudes
                    )
                if peak_index is None:
                    peak_index = temp_peak_index

                freqs = [filtered_freqs[peak_index]]

                mags = [filtered_magnitudes[peak_index]]
                if peak_index > 0:
                    freqs.append(filtered_freqs[peak_index - 1])
                    mags.append(filtered_magnitudes[peak_index - 1])
                if peak_index < len(filtered_freqs) - 1:
                    freqs.append(filtered_freqs[peak_index + 1])
                    mags.append(filtered_magnitudes[peak_index + 1])

                mag_sum = sum(mags)
                heart_rate_bpm = 60 * sum(
                    freq * (mag / mag_sum) for freq, mag in zip(freqs, mags)
                )

                if max_magnitude != 0:
                    filtered_magnitudes = filtered_magnitudes / max_magnitude

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

                timeStamp = time.time()
                heartRate = smoothed_heart_rate

                sliding_window_pulse = sliding_window(
                    sampling_rate, signal, heart_rate_bpm
                )  # rgb value
                smoothedBrightness = (
                    sliding_window_pulse
                    if smoothedBrightness == 0
                    else exp_smooth(
                        smoothedBrightness, sliding_window_pulse, sampling_rate, 20
                    )
                )

                if sliding_window_pulse > smoothedBrightness or (
                    last_pulse_time + (1 / (smoothed_heart_rate / 60)) * 0.8
                    > time.time()
                ):
                    amp_video_brightness = exp_smooth(
                        amp_video_brightness, 0, sampling_rate, 10
                    )
                else:
                    amp_video_brightness = 0.5
                    last_pulse_time = timeStamp
                
                
                brightness_color = int(amp_video_brightness * 255)
                color = (
                    brightness_color,
                    brightness_color,
                    brightness_color,
                )  # Grayscale color

                # Draw a filled rectangle
                cv2.rectangle(
                    frame,
                    (small_forehead_coords[0], small_forehead_coords[1]),
                    (small_forehead_coords[2], small_forehead_coords[3]),
                    color,  # Fill with the calculated brightness
                    thickness=-1,  # -1 thickness fills the rectangle
                )

                result_log.append(
                    {
                        "timestamp": datetime.datetime.fromtimestamp(
                            timeStamp
                        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4]
                        + "Z",
                        "bpm": heartRate,
                        "pulse": amp_video_brightness / 0.5,
                    }
                )
                with open(output_file, "w") as f:
                    json.dump(result_log, f, indent=4)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)

                cv2.putText(
                    frame,
                    f"Heart Rate: {smoothed_heart_rate:.2f} bpm",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    4,
                )

    else:
        cv2.putText(
            frame,
            "No face detected",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4,
        )

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
