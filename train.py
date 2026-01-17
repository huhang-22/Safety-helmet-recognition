from ultralytics import YOLO
import os

def main():
    print("=== 开始YOLOv8安全检测模型训练 ===")
    
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='E:/bishe/trained/data.yaml',  
        epochs=50,              # 训练轮数
        imgsz=640,              # 图像尺寸
        batch=8,                # 批大小
        name='safety_detection_v3',  # 实验名称
        patience=10,            # 早停耐心值
        save=True,              # 保存模型
        device=0            
    )
    
    print("训练完成")
if __name__ == '__main__':
    main()