import numpy as np
import time
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.integrate import quad

# 定义一些常量
CONTOUR_AREA_THRESHOLD, RESIZE_FACTOR = 400, 1
length, dimA, dimB = 50, 50, 50
window_size, slope = 30, 500
m = 0.0325

# Arrays
lower_green = np.array([50, 50, 50])
upper_green = np.array([80, 180, 180])
video_size = np.array([640, 480])

# Lists
Total_energy = []
length_values = []

# String variables
video_path, output_path = "input_path", "output_path"

def insert_depth_32f(depth):
    """
    Apply depth map interpolation using integral images and Gaussian blur.
    """
    height, width = depth.shape
    integral_map = np.zeros((height, width), dtype=np.float64)
    pts_map = np.zeros((height, width), dtype=np.int32)

    # Compute integral images for depth map
    integral_map[depth > 1e-3] = depth[depth > 1e-3]
    pts_map[depth > 1e-3] = 1

    # Compute horizontal integral
    for i in range(height):
        for j in range(1, width):
            integral_map[i, j] += integral_map[i, j - 1]
            pts_map[i, j] += pts_map[i, j - 1]

    # Compute vertical integral
    for i in range(1, height):
        for j in range(width):
            integral_map[i, j] += integral_map[i - 1, j]
            pts_map[i, j] += pts_map[i - 1, j]

    d_wnd = 2.0
    while d_wnd > 1:
        wnd = int(d_wnd)
        d_wnd /= 2

        for i in range(height):
            for j in range(width):
                left = max(j - wnd - 1, 0)
                right = min(j + wnd, width - 1)
                top = max(i - wnd - 1, 0)
                bot = min(i + wnd, height - 1)

                dx = right - left
                dy = (bot - top) * width

                id_left_top = top * width + left
                id_right_top = id_left_top + dx
                id_left_bot = id_left_top + dy
                id_right_bot = id_left_bot + dx

                pts_cnt = pts_map[bot, right] + pts_map[top, left] - (pts_map[bot, left] + pts_map[top, right])
                sum_gray = integral_map[bot, right] + integral_map[top, left] - (
                            integral_map[bot, left] + integral_map[top, right])

                if pts_cnt > 0:
                    depth[i, j] = sum_gray / pts_cnt

        s = wnd // 2 * 2 + 1
        if s > 201:
            s = 201
        depth = cv2.GaussianBlur(depth, (s, s), s)

    return depth


def fill_depth_gaps(depth):
    """
    Fill gaps in depth map with maximum neighbor value if zero.
    """
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i, j] == 0 and 20 < j < depth.shape[1] - 20 and 20 < i < depth.shape[0] - 20:
                b = np.max([
                    depth[i - 20, j],
                    depth[i + 20, j],
                    depth[i, j - 20],
                    depth[i, j + 20]
                ])
                depth[i, j] = b


def midpoint(ptA, ptB):
    """
    Compute midpoint between two points.
    """
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def compute_distance(point1, point2):
    """
    Compute Euclidean distance between two points.
    """
    if np.any(np.isnan(point1)) or np.any(np.isnan(point2)) or np.any(np.isinf(point1)) or np.any(np.isinf(point2)):
        return None  # Return None if any point contains invalid values
    return math.sqrt(np.sum((point1 - point2) ** 2))


def moving_average(data, window_size):
    """
    Compute the moving average of a data series.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def Ft1(s):
    """
    Compute Ft1 based on the input value s using polynomial fitting.
    """
    p1 = 1.008E-5
    p2 = -0.00158
    p3 = 0.18465
    p4 = 0.19057
    alpha = (s - 50) / 50 * 100
    return p1 * alpha ** 3 + p2 * alpha ** 2 + p3 * alpha + p4


def acceleration(x1, x2, x3, x4):
    """
    Compute acceleration based on four points using a fixed time interval.
    """
    t = 1 / 30
    v1 = (x2 - x1) / t
    v2 = (x4 - x3) / t
    return (v2 - v1) / t / 1000


def s(x1, x2):
    """
    Compute the distance in pixels between two points.
    """
    return (x2 - x1) / 1000

# Load calibration data from Excel file
df = pd.read_excel('date7.xlsx', index_col=0, header=None)

# Extract camera matrices and distortion coefficients
left_camera_matrix = np.array(df.iloc[0:3, 0:3], dtype=np.float64)
left_distortion = np.array(df.iloc[5, 0:5], dtype=np.float64).reshape(1, 5)

right_camera_matrix = np.array(df.iloc[6:9, 0:3], dtype=np.float64)
right_distortion = np.array(df.iloc[11, 0:5], dtype=np.float64).reshape(1, 5)

# Extract transformation matrix and rotation matrix
T = np.array(df.iloc[12, 0:3], dtype=np.float64)
R = np.array(df.iloc[13:16, 0:3], dtype=np.float64)

# Set video size
size = (640, 480)  # Example size, replace with video_size if defined elsewhere

# Compute rectification transforms and projection matrices
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    left_camera_matrix, left_distortion,
    right_camera_matrix, right_distortion,
    size, R, T
)

# Compute undistortion and rectification maps
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2
)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2
)
print(Q)

# Load video file
capture = cv2.VideoCapture('video_path')  # Replace with actual video path
capture.set(cv2.CAP_PROP_FPS, 30)
video_name = os.path.splitext(os.path.basename('video_path'))[0]

# Process video frames
fps = 0.0
while True:
    # Start timer
    t1 = time.time()

    # Read a frame from the video
    ret, frame = capture.read()
    if not ret:
        print("Video processing completed")
        capture.release()
        cv2.destroyAllWindows()
        break

    # Split frame into left and right images
    left_frame = frame[0:480, 0:640]
    right_frame = frame[0:480, 640:1280]

    # Resize frames
    left_frame = cv2.resize(left_frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)
    right_frame = cv2.resize(right_frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)

    # Detect green contours in the left frame
    hsv = cv2.cvtColor(left_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(left_frame, left_frame, mask=mask)

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Edge detection
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert to grayscale for rectification
    imgL = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Rectify images
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # Convert rectified images to BGR format
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    blockSize = 5
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=128,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)

    # Initialize stereo matcher
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)  # Example parameters, adjust as needed

    # Compute disparity map
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # Normalize disparity for visualization (gray scale)
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Optionally fill holes in the depth map
    # insert_depth(disp)
    # insertDepth32f(disp)

    # Optionally generate colorized depth map
    # disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    # Optionally apply WLS filtering
    # right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    # left_disp = stereo.compute(img1_rectified, img2_rectified)
    # right_disp = right_matcher.compute(img1_rectified, img2_rectified)
    # wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    # wls_filter.setLambda(300)
    # wls_filter.setSigmaColor(1.2)
    # wls_filter.setLRCthresh(8)
    # wls_filter.setDepthDiscontinuityRadius(3)
    # filtered_disp = wls_filter.filter(left_disp, img1_rectified, disparity_map_right=right_disp)
    # disp = cv2.normalize(filtered_disp, filtered_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Compute 3D coordinates from disparity map
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)

    # Scale 3D coordinates (adjust this factor as needed)
    threeD = threeD * 16

    # Inside the loop
    for c in contours:

        # Initialize variables
        previous_trend, trend_count = None, 0
        peaks, valleys, points = [], [], set()
        previous_length, trend_count_increase, trend_count_decrease = 0, 0, 0
        detect_peaks, detect_valleys = True, False
        energysA, energysB, energy, Force = [], [], [], []

        # Ignore contours smaller than a threshold area
        if cv2.contourArea(c) < CONTOUR_AREA_THRESHOLD:
            continue

        # Get the minimum enclosing rectangle for the contour
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int_(box)

        # Draw the rectangle around the contour on the left frame
        cv2.drawContours(left_frame, [box], 0, (0, 255, 0), 2)

        # Draw circles on the corners of the rectangle
        for point in box:
            cv2.circle(left_frame, tuple(point), 2, (255, 0, 0), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.circle(left_frame, (int(tltrX), int(tltrY)), 2, (255, 0, 0), -1)
        cv2.circle(left_frame, (int(blbrX), int(blbrY)), 2, (255, 0, 0), -1)
        cv2.circle(left_frame, (int(tlblX), int(tlblY)), 2, (255, 0, 0), -1)
        cv2.circle(left_frame, (int(trbrX), int(trbrY)), 2, (255, 0, 0), -1)

        cv2.line(left_frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(left_frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # Get 3D points
        if 0 <= int(tlblY) < threeD.shape[0] and 0 <= int(tlblX) < threeD.shape[1]:
            tltr_3d = threeD[int(tltrY), int(tltrX)]
            blbr_3d = threeD[int(blbrY), int(blbrX)]
            tlbl_3d = threeD[int(tlblY), int(tlblX)]
            trbr_3d = threeD[int(trbrY), int(trbrX)]
        else:
            # Skip if index is out of range
            continue

        # Compute distances dA and dB
        dA = compute_distance(blbr_3d, tltr_3d)
        dB = compute_distance(trbr_3d, tlbl_3d)

        # Set dimensions
        if dA is not None and dB is not None:
            dimA = dA
            dimB = dB
        else:
            dimA = dimA
            dimB = dimB

        # Determine length and width
        length, width = (dimA, dimB) if dimA > dimB else (dimB, dimA)

        detected_valid_size = False

        # Skip size processing until a valid size is detected
        if not detected_valid_size:
            if length != 0 and width != 0:
                detected_valid_size = True
            else:
                continue

        if length <= 300:
            cv2.putText(left_frame, "{:.1f}mm".format(dimA), (int(tltrX - 30), int(tltrY - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(left_frame, "{:.1f}mm".format(dimB), (int(trbrX + 30), int(trbrY)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            pre_length = length
            length_values.append(length)

            # Apply moving average filter
            if len(length_values) >= window_size:

                smoothed_length_values = length_values
                print(smoothed_length_values)

                for frame_num, length in enumerate(smoothed_length_values):
                    frame_num_range = range(frame_num, frame_num + 4)
                    length_change = length - previous_length

                    # Determine the trend of length change
                    if length_change > 0:
                        current_trend = "Increase"
                        trend_count_increase += 1
                        trend_count_decrease = 0
                    elif length_change < 0:
                        current_trend = "Decrease"
                        trend_count_decrease += 1
                        trend_count_increase = 0
                    else:
                        current_trend = "No Change"
                        trend_count_increase = 0
                        trend_count_decrease = 0

                    # Detect peaks and valleys
                    if detect_peaks and trend_count_decrease >= 5:
                        peaks.append((frame_num, length))
                        points.add((frame_num, length))
                        detect_peaks = False
                        detect_valleys = True
                    elif detect_valleys and trend_count_increase >= 5:
                        valleys.append((frame_num, length))
                        points.add((frame_num, length))
                        detect_valleys = False
                        detect_peaks = True

                    # Calculate force and energy
                    if peaks and len(peaks) > len(valleys):
                        Ft = Ft1(length)
                    else:
                        Ft = Ft1(length)

                    if all(0 <= i < len(smoothed_length_values) for i in frame_num_range):
                        # Get four consecutive values
                        consecutive_values = [smoothed_length_values[i] for i in frame_num_range]
                        acc = acceleration(consecutive_values[0], consecutive_values[1], consecutive_values[2],
                                           consecutive_values[3])

                        F = Ft + m * acc
                        s_value = s(consecutive_values[0], consecutive_values[1])
                        energys = F * s_value

                        Force.append(F)
                        energy.append(energys)

                    # Update previous_trend and previous_length
                    previous_trend = current_trend
                    previous_length = length

                # Output detected peaks and valleys
                if peaks:
                    print("Detected peaks:")
                    for peak in peaks:
                        frame_num, length = peak
                        print(f"Peak frame: {frame_num}, Length: {length}")
                else:
                    print("No peaks detected.")

                if valleys:
                    print("Detected valleys:")
                    for valley in valleys:
                        frame_num, length = valley
                        print(f"Valley frame: {frame_num}, Length: {length}")
                else:
                    print("No valleys detected.")

                # Output detected extrema
                if points:
                    print("Detected extrema:")
                    for point in points:
                        frame_num, length = point
                        print(f"Frame: {frame_num}, Length: {length}")
                else:
                    print("No extrema detected.")

                print(Force)
                total_energy = sum(abs(e) for e in energy)
                Total_energy.append(total_energy)
                print(Total_energy)
                N = len(points) * 0.5
                print(f"Total energy: {total_energy}")

                # Plotting
                size = 15
                scale_factor = 30  # Scaling factor
                plt.clf()
                plt.rcParams['font.family'] = 'Times New Roman'
                plt.title('Real-time', fontsize=24)

                # Adjust positions of peaks and valleys
                peaks_x = [(peak[0] + size) / scale_factor for peak in peaks]
                valleys_x = [(valley[0] + size) / scale_factor for valley in valleys]

                # Plot the smoothed_length_values data curve
                plt.plot(np.arange(0, len(smoothed_length_values)) / scale_factor, smoothed_length_values,
                         color='red', label='Length', zorder=2)

                plt.xlabel('Time (s)', fontsize=20)
                plt.ylabel('Length (mm)', fontsize=20, color='red')
                plt.xlim(left=0)
                plt.ylim(bottom=0)

                # Set X-axis ticks
                x_ticks = np.arange(0, len(smoothed_length_values) / scale_factor + 4, 4)
                plt.xticks(x_ticks, fontsize=18)
                plt.yticks(np.arange(0, 101, step=25), fontsize=18)

                # Set color for the left Y-axis
                ax1 = plt.gca()
                ax1.spines['left'].set_color('red')
                ax1.yaxis.label.set_color('red')
                ax1.tick_params(axis='y', colors='red')

                # Add a second Y-axis
                ax2 = ax1.twinx()
                ax2.plot(np.arange(0, len(Force)) / scale_factor, Force, color='blue', label='Force', zorder=1)
                ax2.set_ylabel('Force (N)', fontsize=20, color='blue')
                ax2.set_yticks(np.arange(0, 26, step=6))
                ax2.set_ylim(bottom=0)
                ax2.tick_params(axis='y', labelsize=18, colors='blue')

                # Set color for the right Y-axis
                ax2.spines['right'].set_color('blue')
                ax2.yaxis.label.set_color('blue')

                # Get all legends and labels
                lines_1, labels_1 = ax1.get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(0.64, 0.995),
                           prop={'size': 16})

                plt.tight_layout()  # Adjust layout to prevent overlap

                plt.pause(0.01)

                # Save the plot as an EPS file
                eps_output_path = os.path.join(output_path, f"{video_name}.eps")
                plt.savefig(eps_output_path, format='eps')

                # Save smoothed length values to an Excel file
                output_file_path = os.path.join(output_path, f"{video_name}_smoothed.xlsx")
                dataframe = pd.DataFrame(smoothed_length_values)
                dataframe.to_excel(output_file_path, index=False)
            else:
                print("The list length is insufficient to meet the window size requirement for filtering.")

        # Output to Excel
        length_output_file_path = os.path.join(output_path, f"{video_name}.xlsx")
        length_dataframe = pd.DataFrame(length_values)
        length_dataframe.to_excel(length_output_file_path, index=False)

        total_energy_output_file_path = os.path.join(output_path, f"{video_name}_total_energy.xlsx")
        total_energy_dataframe = pd.DataFrame(Total_energy)
        total_energy_dataframe.to_excel(total_energy_output_file_path, index=False)

        # Display FPS on the video frame
        fps = (fps + (1. / (time.time() - t1))) / 2
        left_frame = cv2.putText(left_frame, "fps= %.2f" % fps, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the video frame
        cv2.imshow("video", left_frame)
        cv2.imshow("depth", disp)

# Release resources and close all windows
capture.release()
cv2.destroyAllWindows()

