import os
import cv2
# import csv
import numpy as np
# import time
import peakutils
import logging
from KeyFrameDetector.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics

logging.basicConfig(
    filename='keyframe_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def keyframe_detect(source, dest, thresh, max_keyframes=8, show_metrics=False, verbose=False):
    keyframe_path = os.path.join(dest, 'keyframes')
    image_grids_path = os.path.join(dest, 'image_grids')
    metrics_path = os.path.join(dest, 'metrics')
    # path2file = os.path.join(csv_path, 'output.csv')
    prepare_dirs(keyframe_path, image_grids_path, metrics_path)

    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # max_sample_frame_num = 640
    max_sample_frame_num = 320
    min_sample_frame_num = 16
    target_frame_num = min(max_sample_frame_num, max(min_sample_frame_num, length))
    # sample_interval = length / target_frame_num
    sample_interval_ms = np.ceil(length / (target_frame_num * fps) * 1000)
    # logging.debug(f'{sample_interval_ms=}')
    current_time = 0

    last_frames = []
    last_diff_mag = []
    # time_spans = []
    images = []
    full_color = []
    last_frame = None
    # start_time = time.process_time()

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time)
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        logging.debug(f'Processing frame {frame_number} of {length}')

        grayframe, blur_gray = convert_frame_to_grayscale(frame)
        last_frames.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)

        if frame_number == 0:
            last_frame = blur_gray
            last_diff_mag.append(0)
            # time_spans.append(0)
            current_time += sample_interval_ms
            continue

        diff = cv2.subtract(blur_gray, last_frame)
        diff_mag = cv2.countNonZero(diff)
        last_diff_mag.append(diff_mag)

        # stop_time = time.process_time()
        # time_spans.append(stop_time - start_time)
        current_time += sample_interval_ms
        last_frame = blur_gray

    cap.release()

    if len(last_diff_mag) < 3:
        logging.warning("Not enough frames for peak detection.")
        return

    y = np.array(last_diff_mag)
    base = peakutils.baseline(y, 2)
    indices = peakutils.indexes(y - base, thresh, min_dist=min(fps, target_frame_num // max_keyframes))

    if len(indices) > max_keyframes:
        ranked_indices = sorted(indices, key=lambda i: last_diff_mag[i], reverse=True)[:max_keyframes]
        indices = sorted(ranked_indices)

    if show_metrics:
        # print(f'{indices=}, {last_frames=}, {last_diff_mag=}')
        plot_metrics(indices, last_frames, last_diff_mag, save_path=os.path.join(metrics_path, 'diff_mag.jpg'))

    cnt = 1
    # write_header = not os.path.exists(path2file)

    for x in indices:
        cv2.imwrite(os.path.join(keyframe_path, f'keyframe_{cnt}.jpg'), full_color[x])
        # log_message = f'keyframe {cnt} happened at {time_spans[x]} sec.'
        # if verbose:
        #     logging.info(log_message)
        # with open(path2file, 'a', newline='') as csv_file:
        #     writer = csv.writer(csv_file)
        #     if write_header:
        #         writer.writerow(["Keyframe Log"])
        #         write_header = False
        #     writer.writerow([log_message])
        cnt += 1

    # cv2.destroyAllWindows()


def frame_extract(source, dest):
    extract_path = os.path.join(dest, 'extract_frames')
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'{length=}')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'{fps=}')

    max_sample_frame_num = 640
    min_sample_frame_num = 16
    target_frame_num = min(max_sample_frame_num, max(min_sample_frame_num, length))
    sample_interval_ms = np.ceil(length / (target_frame_num * fps) * 1000)
    current_time = 0
    print(f'{sample_interval_ms=}')

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time)
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imwrite(os.path.join(extract_path, f"frame@{current_time}ms.jpg"), frame)
        current_time += sample_interval_ms

    cap.release()
    return
