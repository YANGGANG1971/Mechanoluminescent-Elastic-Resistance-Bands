from scipy.spatial import distance as dist
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import pandas as pd


# Scalar values
CONTOUR_AREA_THRESHOLD, RESIZE_FACTOR = 400, 1
object_width, real_length, real_width = 30, 0, 0
window_size, slope = 30, 500
m = 0.0325

# Arrays
lower_green = np.array([45, 40, 40])
upper_green = np.array([80, 200, 200])
video_size = np.array([640, 480])

# Lists
length_values, previous_width, Total_energy = [], [], []

# String variables
video_path, output_path = "input_path", "output_path"

def midpoint(ptA, ptB):

    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def moving_average(data):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def Ft1(s):
    p1 = 1.008E-5
    p2 = -0.00158
    p3 = 0.18465
    p4 = 0.19057
    alpha = (s - 50) / 50 * 100

    return p1 * alpha ** 3 + p2 * alpha ** 2 + p3 * alpha + p4

def acceleration(x1,x2,x3,x4):
    t = 1 / 30
    v1 = ( x2 - x1 ) / t
    v2 = ( x4 - x3 ) / t
    a = ( v2 - v1 ) / t / 1000
    return a

def s(x1,x2):
    s =  (x2 - x1) / 1000
    return s


import cv2
import os
import time

# Load the video file
capture = cv2.VideoCapture(video_path)
capture.set(cv2.CAP_PROP_FPS, 30)
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Initialize frame number and frame rate control at the start of the loop
frame_num = 0
prev_frame_time = time.time()
delay = 1 / 10  # 30 FPS

# Loop through each frame in the video
while True:
    frame_num += 1  # Increment frame count
    t1 = time.time()  # Start timing
    ret, frame = capture.read()  # Check if the frame is read successfully
    if not ret:
        print("Recognition ended")
        capture.release()  # Release resources
        cv2.destroyAllWindows()  # Close all windows
        break

    # Split the frame into left and right images
    left_frame = frame[0:480, 0:640]
    right_frame = frame[0:480, 640:1280]

    # Resize the left frame
    left_frame = cv2.resize(left_frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)

    # Convert to HSV color space
    hsv = cv2.cvtColor(left_frame, cv2.COLOR_BGR2HSV)

    # Create a mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply bitwise AND to get the result
    result = cv2.bitwise_and(left_frame, left_frame, mask=mask)

    # Convert to grayscale
    gray = cv2.GaussianBlur(result, (7, 7), 0)

    # Canny edge detection
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pixels per metric ratio
    pixelsPerMetric = None

    for c in contours:

        # Initialize variables
        previous_trend, trend_count = None, 0
        peaks, valleys, points = [], [], set()
        previous_length, trend_count_increase, trend_count_decrease = 0, 0, 0
        detect_peaks, detect_valleys = True, False
        energysA, energysB, energy, Force = [], [], [], []

        # Ignore contours smaller than a certain area
        if cv2.contourArea(c) < CONTOUR_AREA_THRESHOLD:
            continue

        # Get the minimum enclosing rectangle for the contour
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Draw the rectangle contour on the left image
        cv2.drawContours(left_frame, [box], 0, (0, 255, 0), 2)

        # Draw circles at the corners of the rectangle
        for point in box:
            cv2.circle(left_frame, tuple(point), 2, (255, 0, 0), -1)

        # Calculate midpoints of the sides of the rectangle
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # Draw circles at the midpoints
        cv2.circle(left_frame, (int(tltrX), int(tltrY)), 2, (255, 0, 0), -1)
        cv2.circle(left_frame, (int(blbrX), int(blbrY)), 2, (255, 0, 0), -1)
        cv2.circle(left_frame, (int(tlblX), int(tlblY)), 2, (255, 0, 0), -1)
        cv2.circle(left_frame, (int(trbrX), int(trbrY)), 2, (255, 0, 0), -1)

        # Draw lines between the midpoints
        cv2.line(left_frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(left_frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # Calculate the heights and widths
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # Determine length and width
        length, width = (dA, dB) if dA > dB else (dB, dA)

        detected_valid_size = False

        # Skip size processing until a valid size is detected
        if not detected_valid_size:
            if length != 0 and width != 0:
                detected_valid_size = True
            else:
                continue

        # Update previous width list if a valid size is detected
        if detected_valid_size:
            if len(previous_width) < 20:
                previous_width.append(width)
            else:
                # Calculate average width and pixels per metric
                ave_width = sum(previous_width) / 20
                pixels_per_metric = ave_width / object_width
                print(pixels_per_metric)

                if pixels_per_metric is not None:
                    real_length = length / pixels_per_metric
                    real_width = width / pixels_per_metric

                    # Display real dimensions if within valid range
                    if real_length <= 300:
                        cv2.putText(left_frame, "{:.1f}mm".format(real_length),
                                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (255, 255, 255), 2)
                        cv2.putText(left_frame, "{:.1f}mm".format(real_width),
                                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (255, 255, 255), 2)

                        length_values.append(real_length)
                        print(f"Length: {real_length:.2f}, Width: {real_width:.2f}")
                        print(length_values)

                        # Apply moving average filter if enough values are collected
                        if len(length_values) >= window_size:
                            smoothed_length_values = moving_average(length_values).tolist()
                            print(smoothed_length_values)

                            for idx, smoothed_length in enumerate(smoothed_length_values):
                                idx_range = range(idx, idx + 4)

                                # Calculate length change
                                length_change = smoothed_length - previous_length

                                # Determine the trend based on length change
                                if length_change > 0:
                                    current_trend = "Increasing"
                                    trend_count_increase += 1
                                    trend_count_decrease = 0
                                elif length_change < 0:
                                    current_trend = "Decreasing"
                                    trend_count_decrease += 1
                                    trend_count_increase = 0
                                else:
                                    current_trend = "Stable"
                                    trend_count_increase = 0
                                    trend_count_decrease = 0

                                # Detect peaks
                                if detect_peaks and trend_count_decrease >= 5:
                                    peaks.append((idx, smoothed_length))
                                    points.add((idx, smoothed_length))
                                    detect_peaks = False
                                    detect_valleys = True

                                # Detect valleys
                                elif detect_valleys and trend_count_increase >= 5:
                                    valleys.append((idx, smoothed_length))
                                    points.add((idx, smoothed_length))
                                    detect_valleys = False
                                    detect_peaks = True

                                # Calculate force
                                Ft = Ft1(smoothed_length) if len(peaks) <= len(valleys) else Ft1(smoothed_length)

                                # Compute acceleration and force if sufficient values are available
                                if all(0 <= i < len(smoothed_length_values) for i in idx_range):
                                    consecutive_values = [smoothed_length_values[i] for i in idx_range]
                                    acc = acceleration(consecutive_values[0], consecutive_values[1],
                                                       consecutive_values[2], consecutive_values[3])

                                    Fn = Ft + m * acc
                                    s_value = s(consecutive_values[0], consecutive_values[1])
                                    F = Fn * 2
                                    energys = F * s_value

                                    Force.append(F)
                                    energy.append(energys)

                                # Update previous trend and length
                                previous_trend = current_trend
                                previous_length = smoothed_length

                            # Print detected peaks and valleys
                            if peaks:
                                print("Detected Peaks:")
                                for peak in peaks:
                                    idx, smoothed_length = peak
                                    print(f"Peak Frame: {idx}, Length: {smoothed_length}")
                            else:
                                print("No Peaks Detected.")

                            if valleys:
                                print("Detected Valleys:")
                                for valley in valleys:
                                    idx, smoothed_length = valley
                                    print(f"Valley Frame: {idx}, Length: {smoothed_length}")
                            else:
                                print("No Valleys Detected.")

                            # Print force values
                            print(Force)

                            # Calculate total energy
                            total_energy = sum(energy)
                            Total_energy.append(total_energy)
                            N = len(points) * 0.5
                            print(f"Total Energy: {total_energy}")

                    # Processing data
                    size = 15
                    scale_factor = 60  # Scaling factor
                    plt.clf()
                    plt.rcParams['font.family'] = 'Times New Roman'
                    plt.title('Real-time', fontsize=24)

                    # Plot the Force data curve
                    plt.plot(np.arange(0, len(Force)) / scale_factor, Force, color='blue', label='Force', zorder=2)

                    # Plot other data (if any)
                    # plt.plot(np.arange(len(length_values)) / scale_factor, length_values, color='gray', label='Length')

                    plt.xlabel('Time (s)', fontsize=20)  # X-axis label font size
                    plt.ylabel('Force (N)', fontsize=20)  # Left Y-axis label font size
                    plt.xlim(left=0)
                    plt.ylim(bottom=0)

                    # Set X-axis ticks
                    x_ticks = np.arange(0, len(smoothed_length_values) / scale_factor + 4, 4)
                    plt.xticks(x_ticks, fontsize=18)
                    plt.yticks(np.arange(0, 26, step=6), fontsize=18)

                    # Set color for the left Y-axis
                    ax1 = plt.gca()
                    ax1.spines['left'].set_color('blue')
                    ax1.yaxis.label.set_color('blue')
                    ax1.tick_params(axis='y', colors='blue')

                    # Add a second Y-axis
                    ax2 = ax1.twinx()
                    ax2.plot(np.arange(0, len(Total_energy)) / scale_factor, Total_energy, color='green',
                             linestyle='--', label='Total Work', zorder=1)
                    ax2.set_ylabel('Total Work (J)', fontsize=18)  # Right Y-axis label font size
                    ax2.set_yticks(np.arange(-0.12, 0.28, step=0.09))
                    ax2.set_ylim(bottom=-0.03)
                    ax2.tick_params(axis='y', labelsize=18)

                    # Set color for the right Y-axis
                    ax2.spines['right'].set_color('green')
                    ax2.yaxis.label.set_color('green')
                    ax2.tick_params(axis='y', colors='green')

                    # Get all legends and labels
                    lines_1, labels_1 = ax1.get_legend_handles_labels()
                    lines_2, labels_2 = ax2.get_legend_handles_labels()
                    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(0.56, 0.995),
                               prop={'size': 16})
                    plt.tight_layout()  # Adjust layout to prevent overlap

                    # Remove borders
                    # plt.gca().spines['top'].set_visible(False)
                    # plt.gca().spines['right'].set_visible(False)

                    plt.pause(0.01)

                    # Save the plot as an EPS file
                    eps_output_path = os.path.join(output_path, f"{video_name}.eps")
                    plt.savefig(eps_output_path, format='eps')

                    # Save smoothed length values to an Excel file
                    smoothed_output_file_path = os.path.join(output_path, f"{video_name}_smoothed.xlsx")
                    smoothed_dataframe = pd.DataFrame(smoothed_length_values)
                    smoothed_dataframe.to_excel(smoothed_output_file_path, index=False)
                else:
                    print("The list length is insufficient to meet the window size requirement for filtering.")

                # Save length values to Excel
                output_file_path = os.path.join(output_path, f"{video_name}.xlsx")
                dataframe = pd.DataFrame(length_values)
                dataframe.to_excel(output_file_path, index=False)

                # Update FPS
                current_time = time.time()
                elapsed_time = current_time - prev_frame_time
                fps = 1 / elapsed_time
                prev_frame_time = current_time

                fps = (fps + (1. / (time.time() - t1))) / 2
                left_frame = cv2.putText(left_frame, f"fps= {fps:.2f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                         (0, 255, 0), 2)
                cv2.imshow("video", left_frame)

# Release resources and close all windows
capture.release()
cv2.destroyAllWindows()