from ultralytics import YOLO

def main():
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 可选: yolov8s/m/l/x.pt
    
    # 训练（自动使用GPU）
    results = model.train(
        data='data.yaml',
        epochs=50,           # 训练轮数
        imgsz=640,           # 图像尺寸
        batch=16,            # 批次大小（根据显存调整）
        device='0',          # GPU设备（'0' 或 '0,1' 多卡）
        workers=4,           # 数据加载线程
        lr0=0.01,            # 初始学习率
        augment=True,        # 数据增强
        val=True             # 启用验证
    )
    
    print(f"训练完成！结果保存于: {results.save_dir}")

if __name__ == "__main__":
    main()
    
