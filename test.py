# test_model.py
from ultralytics import YOLO
import os
import cv2
import glob

def main():
    print("使用训练好的模型进行测试")

    model_path = 'runs/detect/safety_detection_v3/weights/best.pt'
    
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    

    test_images_dir = 'trained/test/images/'  
    # image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    # image_files = []
    # for ext in image_extensions:
    #     image_files.extend(glob.glob(os.path.join(test_images_dir, ext)))
    test_source =test_images_dir # image_files[:30]
    
    print("开始预测")
    results = model.predict(
        source=test_source,
        conf=0.25,      # 置信度阈值（大于此值的检测框才会显示）
        iou=0.45,       # NMS的IoU阈值
        save=True,      # 保存标注后的图片
        save_txt=False, # 不保存标签文本文件（设置为True可保存检测框坐标）
        save_conf=True, # 在保存的图片上显示置信度
        show_labels=True, # 显示标签
        show_conf=True,  # 显示置信度
        max_det=50,     # 每张图片最大检测数量
        project='runs/detect',  # 结果保存的主目录
        name='predict', # 结果保存的子目录名称
        exist_ok=True   # 允许覆盖已存在的预测结果
    )

    print(f"预测完成")
    
    save_dir = 'runs/detect/predict'
    # if os.path.exists(save_dir):
    #     print(f"标注好的图片保存在: {os.path.abspath(save_dir)}")
    #     image_files = [f for f in os.listdir(save_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #     if image_files:
    #         print(f"共生成 {len(image_files)} 张标注图片:")
    #         for i, img_file in enumerate(image_files[:5]):  # 只显示前5个
    #             print(f"  {i+1}. {img_file}")
    #         if len(image_files) > 5:
    #             print(f"  ... 以及 {len(image_files)-5} 张更多图片")
    #     else:
    #         print("警告：未找到保存的图片文件")

    # if results and len(results) > 0:
    #     print(f"\n 检测统计:")
    #     total_detections = 0
    #     for i, r in enumerate(results):
    #         if hasattr(r, 'boxes') and r.boxes is not None:
    #             num_detections = len(r.boxes)
    #             total_detections += num_detections
    #             print(f"  图片{i+1}: 检测到 {num_detections} 个目标")
        
    #     print(f"  总计: {total_detections} 个检测目标")

    #     if hasattr(results[0], 'names'):
    #         class_names = results[0].names
    #         class_counts = {}
            
    #         for r in results:
    #             if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
    #                 for cls in r.boxes.cls:
    #                     class_id = int(cls)
    #                     class_name = class_names.get(class_id, f'class_{class_id}')
    #                     class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
    #         if class_counts:
    #             print(f"\n 类别分布:")
    #             for class_name, count in class_counts.items():
    #                 print(f"  {class_name}: {count} 个")

if __name__ == '__main__':
    main()