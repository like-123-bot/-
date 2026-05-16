# 交通标志检测实验报告

## 作者信息
- **姓名**: 李可
- **学号**: 112305010332

---

## 任务描述
本实验旨在训练一个目标检测模型，对交通标志数据集进行检测。使用 YOLOv8 作为基础模型，完成以下任务：
1. 在训练集上训练模型
2. 在验证集上评估模型性能
3. 对测试集进行预测并生成提交文件

---

## 数据集

### 类别列表（15类）
| ID | 类别名称 | ID | 类别名称 |
|---|---------|---|---------|
| 0 | Green Light | 8 | Speed Limit 40 |
| 1 | Red Light | 9 | Speed Limit 50 |
| 2 | Speed Limit 10 | 10 | Speed Limit 60 |
| 3 | Speed Limit 100 | 11 | Speed Limit 70 |
| 4 | Speed Limit 110 | 12 | Speed Limit 80 |
| 5 | Speed Limit 120 | 13 | Speed Limit 90 |
| 6 | Speed Limit 20 | 14 | Stop |
| 7 | Speed Limit 30 | | |

### 数据结构
├── train/ # 训练集 │ ├── images/ # 训练图像 │ └── labels/ # 训练标签（YOLO格式） ├── val/ # 验证集 │ ├── images/ # 验证图像 │ └── labels/ # 验证标签 └── test/ # 测试集 └── images/ # 测试图像（无标签）


plainText

### 标签格式
YOLO 格式（归一化坐标）：
<class_id> <x_center> <y_center>



plainText

---

## 环境配置

### 依赖安装
```bash
pip install ultralytics
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 设备要求
- **GPU**: NVIDIA RTX 4060 或更高
- **显存**: 8GB 或更高
- **CUDA**: 12.1 或更高

---

## 训练步骤

### 训练命令
```bash
# YOLOv8n
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16 device=0

# YOLOv8s（更高精度）
yolo detect train data=data.yaml model=yolov8s.pt epochs=50 imgsz=640 batch=8 device=0
```

### 训练参数说明
| 参数 | 说明 | 值 |
|-----|------|---|
| epochs | 训练轮数 | 50 |
| imgsz | 图像尺寸 | 640 |
| batch | 批次大小 | 16/8 |
| device | GPU设备 | 0 |

### 训练结果
模型保存于 `runs/detect/train/weights/best.pt`

---

## 推理步骤

### 生成提交文件
```bash
python baseline_infer.py --model runs/detect/train/weights/best.pt --test-dir test/images --output submission.csv --conf 0.25
```

### 提交文件格式
```csv
image_id,class_id,x_center,y_center,width,height,confidence
000003_jpg.rf.xxx.jpg,7,0.500,0.480,0.455,0.609,0.858
```

---

## 评估指标

### 训练指标
| 指标 | 说明 | 目标值 |
|-----|------|-------|
| mAP@0.5 | IoU=0.5时的平均精度 | 越高越好 |
| box_loss | 边界框损失 | 越低越好 |
| cls_loss | 分类损失 | 越低越好 |

### 实验结果
- **训练轮数**: 50
- **mAP@0.5**: ~0.57
- **训练时间**: ~25分钟（RTX 4060）

---

## 项目文件结构
. ├── data.yaml # 数据集配置 ├── train.py # 训练脚本 ├── baseline_infer.py # 推理脚本 ├── train/ # 训练集 ├── val/ # 验证集 ├── test/ # 测试集 └── runs/ # 训练结果 └── detect/train/weights/best.pt


plainText

---

## 注意事项
1. 确保训练前已准备好 `train/` 和 `val/` 目录
2. 标签文件需与图像文件同名（.txt后缀）
3. 提交文件的 image_id 必须与测试集文件名一致
4. 坐标值需归一化到 [0, 1] 范围

---

## 参考文献
1. YOLOv8 官方文档: https://docs.ultralytics.com/
2. Ultralytics GitHub: https://github.com/ultralytics/ultral
