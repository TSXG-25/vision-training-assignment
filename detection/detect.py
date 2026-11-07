from pathlib import Path
import argparse
import yaml
import cv2
import numpy as np
import openvino as ov
import json



def corners_to_box(corners):
    x_min = np.min(corners[:, 0])
    y_min = np.min(corners[:, 1])

    x_max = np.max(corners[:, 0])
    y_max = np.max(corners[:, 1])

    return np.array([x_min, y_min, x_max, y_max])

def calculate_iou(box_a, box_b):

    # 计算重叠矩形
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])

    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    # 计算重叠宽高
    width = max(0, x_right - x_left)
    height = max(0, y_bottom - y_top)

    intersection = width * height

    # 分别计算两个框的面积
    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )

    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    # 并集
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union

def nms(boxes, scores, threshold=0.5):

    # 置信度从大到小排序
    indices = np.argsort(scores)[::-1]

    keep = []

    while len(indices) > 0:
        # 取出最高置信度候选框的索引
        current = indices[0]
        # 将 current 加入 keep
        keep.append(current)
        remaining = []
        # 遍历剩余候选框
        for index in indices[1:]:

            iou = calculate_iou(
                boxes[current],
                boxes[index]
            )

            # 只有 IoU <= threshold 的框才保留到下一轮
            if iou <= threshold:
                remaining.append(index)

        indices = np.array(remaining, dtype=int)

    return keep


def restore_corners(corners, scale, pad_x, pad_y):

    original = corners.copy()
    original[:, 0] = (original[:, 0] - pad_x) / scale
    original[:, 1] = (original[:, 1] - pad_y) / scale

    return original


def get_object_points(pattern):
    if pattern in ("1", "base_big"):
        width = 0.230
    else:
        width = 0.135

    height = 0.056

    return np.array([
        [-width / 2, -height / 2, 0],
        [-width / 2,  height / 2, 0],
        [ width / 2,  height / 2, 0],
        [ width / 2, -height / 2, 0]
    ], dtype=np.float64)

def estimate_pose(original_corners, object_points, camera_matrix, dist_coeffs):

    # 获取多个候选位姿
    retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points,
        original_corners.astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    if retval == 0:
        return None

    best_result = None
    best_rmse = float("inf")
    count = 0
    # 检查每个候选
    for rvec, tvec in zip(rvecs, tvecs):
        count += 1
        #print(f"Checking Pose {count}")

        R, _ = cv2.Rodrigues(rvec)

        corners_camera = (R @ object_points.T + tvec).T

        # 检查坐标是否有效
        if not np.all(np.isfinite(corners_camera)):
            continue

        # 检查所有角点是否都在相机前方
        if np.any(corners_camera[:, 2] <= 0):
            continue

        # Step 3: 重新投影
        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )

        projected_points = projected_points.reshape(4, 2)

        errors = projected_points - original_corners
        squared_errors = np.sum(errors**2, axis=1)
        rmse = np.sqrt(np.mean(squared_errors))
        #print(f"Candidate {count}: RMSE = {rmse:.4f} px")
        if not np.isfinite(rmse):
            continue

        if best_rmse > rmse:
            best_rmse = rmse
            best_result = (corners_camera, rmse)

    return best_result


def process_frame(frame, compiled_model, camera_matrix, dist_coeffs):
    detections = []
    # 保持长宽比，缩放到 640×640
    height, width = frame.shape[:2]

    scale = min(640 / width, 640 / height)

    new_w = round(width * scale)
    new_h = round(height * scale)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (new_w, new_h))

    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)

    pad_x = (640 - new_w) // 2
    pad_y = (640 - new_h) // 2

    canvas[
        pad_y:pad_y + new_h,
        pad_x:pad_x + new_w
    ] = resized

    # RGB -> NCHW -> FP16
    tensor = canvas.transpose(2, 0, 1)[None]
    tensor = tensor.astype(np.float16) / np.float16(255.0)

    #print("Input tensor:", tensor.shape, tensor.dtype)

    # 模型推理
    result = compiled_model({
        compiled_model.input(0): tensor
    })

    output = result[compiled_model.output(0)]

    #print("Output tensor:", output.shape)

    # 检查检测置信度
    predictions = output[0]
    objectness_logits = predictions[:, 8]
    scores = 1.0 / (
        1.0 + np.exp(-np.clip(objectness_logits, -30, 30))
    )

    #置信度筛选
    mask = scores > 0.5
    filtered_predictions = predictions[mask]
    filtered_scores = scores[mask]

    # 提取所有角点
    corners = filtered_predictions[:, :8].reshape(-1, 4, 2)

    # 角点转换成 Bounding Boxes
    boxes = np.array([corners_to_box(corner) for corner in corners], dtype=np.float32).reshape(-1, 4)
    #print("Before NMS:", len(boxes))

    # NMS
    keep = nms(boxes, filtered_scores, threshold=0.5)
    final_predictions = filtered_predictions[keep]
    final_scores = filtered_scores[keep]
    #print("After NMS:", len(final_predictions))


    patterns_name = ["sentry", "1", "2", "3", "4", "5", "outpost", "base_small", "base_big"]
    colors_name = ["blue", "red", "neutral", "purple"]
    for prediction, score in zip(final_predictions, final_scores):
        corners = prediction[:8].reshape(4, 2)
        original_corners = restore_corners(corners, scale, pad_x, pad_y)
        
        # 获得读取量
        objectness = prediction[8]
        colors = prediction[9:13]
        patterns = prediction[13:22]

        color_id = int(np.argmax(colors))
        color = colors_name[color_id] #颜色
        pattern_id = np.argmax(patterns)
        pattern = patterns_name[pattern_id] #pattern

        object_points = get_object_points(pattern)

        result = estimate_pose(
            original_corners,
            object_points,
            camera_matrix,
            dist_coeffs
        )

        if result is not None:
            corners_camera, rmse = result

            detection = {
                "color": color,
                "pattern": pattern,
                "confidence": float(score),
                "corners_px": original_corners.tolist(),
                "corners_camera_m": corners_camera.tolist(),
                "reprojection_error_px": float(rmse)
            }

        else:
            detection = {
                "color": color,
                "pattern": pattern,
                "confidence": float(score),
                "corners_px": original_corners.tolist(),
                "corners_camera_m": None,
                "reprojection_error_px": None
            }

        detections.append(detection)

    return detections


def main():

    parser = argparse.ArgumentParser(
        description="Armor plate detection and pose estimation"
    )

    parser.add_argument(
    "--model",
    type=Path,
    required=True,
    help="Path to ONNX model"
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video"
    )

    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Path to camera calibration YAML"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSONL"
)

    args = parser.parse_args()

    MODEL_PATH = args.model
    VIDEO_PATH = args.video
    CALIBRATION_PATH = args.calibration
    OUTPUT_PATH = args.output

    # 检查资源
    assert MODEL_PATH.is_file(), f"Missing model: {MODEL_PATH}"
    assert VIDEO_PATH.is_file(), f"Missing video: {VIDEO_PATH}"
    assert CALIBRATION_PATH.is_file(), f"Missing calibration: {CALIBRATION_PATH}"

    # 加载模型
    core = ov.Core()
    model = core.read_model(str(MODEL_PATH))
    compiled_model = core.compile_model(model, "CPU")

    #print("Model input:", compiled_model.input(0))
    #print("Model output:", compiled_model.output(0))

    #读取camera畸变
    with CALIBRATION_PATH.open("r", encoding="utf-8") as f:
        calibration = yaml.safe_load(f)
    camera_matrix = np.array(
        calibration["camera_matrix"],
        dtype=np.float64
    )
    dist_coeffs = np.array(
        calibration["distort_coeffs"],
        dtype=np.float64
    )

    
    #print("Camera matrix:")
    #print(camera_matrix)

    #print("Distortion coefficients:")
    #print(dist_coeffs)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS: {fps}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    # 流式输入文件
    try:
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                detections = process_frame(
                    frame,
                    compiled_model,
                    camera_matrix,
                    dist_coeffs
                )
                frame_result = {
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / fps,
                    "detections": detections
                }
                f.write(
                    json.dumps(
                        frame_result,
                        ensure_ascii=False,
                        allow_nan=False
                    ) + "\n"
                )
                frame_index += 1
                if frame_index % 30 == 0:
                    print(f"Processed {frame_index} frames")
    finally:
        cap.release()

    print(f"Finished: {frame_index} frames")
    print(f"Output: {OUTPUT_PATH}")

    
if __name__ == "__main__":        
    main()



