from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--test-dir", default="test/images", help="Directory of test images")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    args = parser.parse_args()

    model = YOLO(args.model)
    
    # 获取测试目录中的所有图片文件
    test_dir = Path(args.test_dir)
    image_paths = sorted([p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    
    print(f"找到 {len(image_paths)} 张测试图片")

    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        
        # 对每张图片进行预测
        for img_path in image_paths:
            result = model.predict(source=str(img_path), conf=args.conf, save=False, verbose=False)[0]
            image_id = img_path.name  # 使用实际文件名
            
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                x_center, y_center, width, height = box.xywhn[0].tolist()
                writer.writerow({
                    "image_id": image_id,
                    "class_id": int(box.cls[0].item()),
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "confidence": float(box.conf[0].item()),
                })
    
    print(f"提交文件已生成: {args.output}")


if __name__ == "__main__":
    main()
