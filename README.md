# 装甲板检测与三维位姿估计

## 1. 项目简介

本项目基于提供的 ONNX 装甲板检测模型，实现视频中的装甲板检测、分类以及三维位姿估计。

程序使用 OpenVINO 进行模型推理，结合置信度筛选、非极大值抑制（NMS）和 PnP 位姿求解，获得装甲板在原始图像中的二维角点，以及相机坐标系下的三维角点。

最终结果以 JSONL 格式逐帧输出。

## 2. 实现流程

整体处理流程如下：

```text
加载 ONNX 模型与相机标定参数
            ↓
       读取视频帧
            ↓
    Letterbox 图像预处理
            ↓
      OpenVINO 模型推理
            ↓
        置信度筛选
            ↓
      NMS 去除重复检测
            ↓
   恢复原始图像二维角点
            ↓
    识别装甲板颜色与编号
            ↓
   根据类别选择装甲板尺寸
            ↓
      PnP 三维位姿估计
            ↓
     检查多个候选位姿
            ↓
       计算重投影误差
            ↓
    选择误差最小的有效位姿
            ↓
     转换为相机三维坐标
            ↓
       写入 JSONL 文件
```

### 2.1 模型推理与预处理

首先加载 ONNX 检测模型，并使用 OpenVINO 在 CPU 上执行推理。

每一帧图像进行以下预处理：

1. 将 BGR 图像转换为 RGB。
2. 使用 Letterbox 保持原始长宽比，将图像缩放并填充到 640×640。
3. 将像素值归一化到 `[0,1]`。
4. 转换为 FP16、NCHW 格式，输入模型。

模型输出形状为 `(1, 25200, 22)`，每个候选包含：

- 8 个数值：装甲板四个二维角点。
- 1 个 Objectness Logit。
- 4 个颜色分类 Logits。
- 9 个图案分类 Logits。

### 2.2 置信度筛选与 NMS

首先对 Objectness Logit 应用 Sigmoid：

\[
confidence=\frac{1}{1+e^{-logit}}
\]

保留置信度大于 `0.5` 的候选装甲板。

随后使用非极大值抑制（NMS）去除重复检测：

1. 根据四个预测角点计算轴对齐的 Bounding Box。
2. 按置信度从高到低排列候选框。
3. 优先保留置信度最高的候选。
4. 计算该候选与剩余候选的 IoU。
5. 抑制 IoU 大于 `0.5` 的重叠候选。
6. 对剩余候选重复上述操作。

最终获得经过筛选的装甲板检测结果。

### 2.3 二维角点恢复与类别识别

模型预测的角点位于 640×640 的 Letterbox 图像坐标系中。

通过逆变换恢复原始视频中的像素坐标：

\[
x=\frac{x'-pad_x}{scale}
\]

\[
y=\frac{y'-pad_y}{scale}
\]

随后根据颜色和图案分类 Logits 的最大值对应索引，确定装甲板的颜色和编号。

四个角点按照以下顺序保存：

1. 左上
2. 左下
3. 右下
4. 右上

这里的角点顺序指装甲板自身的角点对应关系，不会在 PnP 前重新按屏幕坐标排序。

### 2.4 PnP 三维位姿估计

利用相机标定参数、装甲板的已知物理尺寸，以及模型检测得到的四个二维角点，进行三维位姿估计。

不同类型装甲板使用不同尺寸：

| 类型 | 宽度 | 高度 |
|---|---|---|
| 普通装甲板 | 0.135 m | 0.056 m |
| 大装甲板（`1`、`base_big`） | 0.230 m | 0.056 m |

以装甲板中心为局部坐标系原点，建立四个三维角点。

由于四个角点位于同一平面，使用 OpenCV 的 `solvePnPGeneric()`，并指定 `SOLVEPNP_IPPE` 方法获取候选位姿。

每个位姿包含：

- `rvec`：旋转向量。
- `tvec`：平移向量。

通过 Rodrigues 变换获得旋转矩阵 \(R\)，再根据：

\[
P_{camera}=RP_{object}+t
\]

将装甲板局部坐标转换为相机坐标系下的三维坐标。

### 2.5 重投影误差与位姿选择

PnP 可能返回多个候选位姿，因此需要进一步检查并选择。

对于每个候选位姿：

1. 检查计算结果是否包含无效数值。
2. 检查四个三维角点的深度是否均为正数。
3. 使用 `cv2.projectPoints()` 将三维角点重新投影到二维图像。
4. 比较重投影角点与模型检测角点的差异。
5. 计算重投影均方根误差（RMSE）。

计算公式：

\[
RMSE=
\sqrt{
\frac{1}{4}
\sum_{i=1}^{4}
\left\|
p_i^{projected}-p_i^{detected}
\right\|^2
}
\]

最终选择所有有效候选中 RMSE 最小的位姿，并输出对应的四个三维角点。

如果 PnP 无法获得有效位姿，则保留二维检测结果，将三维坐标与重投影误差设为 `null`。

## 3. 运行环境

项目使用 Python 3.11 开发。

主要依赖：

- NumPy
- OpenCV
- OpenVINO
- PyYAML

安装依赖：

```bash
pip install numpy opencv-python openvino pyyaml
```

## 4. 使用方法

进入 `detection` 目录后执行。

### Small 视频

```bash
python detect.py \
  --model assets/0526.onnx \
  --video assets/small.mp4 \
  --calibration assets/calibration.yaml \
  --output small.jsonl
```

### Large 视频

```bash
python detect.py \
  --model assets/0526.onnx \
  --video assets/large.mp4 \
  --calibration assets/calibration.yaml \
  --output large.jsonl
```

### 命令行参数

| 参数 | 说明 |
|---|---|
| `--model` | ONNX 模型路径 |
| `--video` | 输入视频路径 |
| `--calibration` | 相机标定 YAML 文件路径 |
| `--output` | 输出 JSONL 文件路径 |

四个参数均为必填项。

## 5. 输出格式

程序逐帧输出 JSONL 文件，每一行对应一个视频帧。

每帧包含以下字段：

| 字段 | 含义 |
|---|---|
| `frame_index` | 当前帧索引，从 0 开始 |
| `timestamp_s` | 当前帧时间戳，单位为秒 |
| `detections` | 当前帧的装甲板检测结果列表 |

每个检测结果包含：

| 字段 | 含义 |
|---|---|
| `color` | 装甲板颜色 |
| `pattern` | 装甲板编号或类型 |
| `confidence` | 检测置信度 |
| `corners_px` | 四个二维角点，单位为像素 |
| `corners_camera_m` | 相机坐标系下的四个三维角点，单位为米 |
| `reprojection_error_px` | PnP 重投影误差，单位为像素 |

示例（仅展示格式，不代表实际检测数据）：

```json
{
  "frame_index": 0,
  "timestamp_s": 0.0,
  "detections": [
    {
      "color": "red",
      "pattern": "3",
      "confidence": 0.98,
      "corners_px": [
        [600.0, 400.0],
        [600.0, 450.0],
        [720.0, 450.0],
        [720.0, 400.0]
      ],
      "corners_camera_m": [
        [-0.0675, -0.028, 3.0],
        [-0.0675, 0.028, 3.0],
        [0.0675, 0.028, 3.0],
        [0.0675, -0.028, 3.0]
      ],
      "reprojection_error_px": 0.21
    }
  ]
}
```

如果当前帧没有检测到装甲板，仍然输出对应帧记录，`detections` 为空列表：

```json
{
  "frame_index": 1,
  "timestamp_s": 0.03333333333333333,
  "detections": []
}
```

## 6. 实现说明

本项目使用提供的预训练模型进行推理，没有重新训练检测模型。

主要实现内容包括：

- OpenVINO 模型推理与预处理。
- 基于 Objectness 的置信度筛选。
- IoU 计算与 Greedy NMS。
- Letterbox 坐标逆变换。
- 装甲板颜色、图案解码与物理尺寸选择。
- 基于 IPPE 的平面 PnP 位姿估计。
- 多候选位姿有效性检查与 RMSE 比较。
- 全视频逐帧处理和 JSONL 流式输出。
