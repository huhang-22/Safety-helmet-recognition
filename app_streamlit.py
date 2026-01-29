# app_streamlit.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# 页面设置
st.set_page_config(
    page_title="工厂安全检测系统",
    page_icon="1",
    layout="wide"
)

# 标题
st.title("工厂安全帽检测系统")
st.markdown("上传工厂场景图片，自动检测工人安全防护装备佩戴情况")

# 侧边栏
with st.sidebar:
    st.header("设置")
    
    # 置信度阈值滑块
    confidence = st.slider(
        "置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        help="值越高，检测要求越严格"
    )
    
    # 模型选择
    model_option = st.selectbox(
        "选择模型",
        ["最佳模型 (best.pt)", "最后模型 (last.pt)"]
    )
    
    # 显示类别说明
    st.header("检测情况")
    st.markdown("""
    - **helmet**: 佩戴安全帽
    - **no-helmet**: 未佩戴安全帽
    """)

# 主界面
col1, col2 = st.columns(2)

with col1:
    st.header("上传图片")
    
    # 图片上传
    uploaded_file = st.file_uploader(
        "选择一张工厂场景图片",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        # 显示原图
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 检测按钮
        if st.button("开始检测", type="primary"):
            with st.spinner("检测中..."):
                # 加载模型
                model_path = 'runs/detect/safety_detection_v3/weights/best.pt'
                model = YOLO(model_path)
                
                # 进行预测
                results = model.predict(
                    source=img_array,
                    conf=confidence,
                    save=False
                )
                
                # 获取结果
                result = results[0]
                
                # 绘制检测结果
                result_img = result.plot()
                
                with col2:
                    st.header(" 检测结果")
                    
                    # 显示结果图片
                    st.image(result_img, caption="检测结果", use_container_width=True)
                    
                    # 显示统计信息
                    if result.boxes is not None:
                        num_detections = len(result.boxes)
                        st.success(f" 检测到 {num_detections} 个目标")
                        
                        # 统计各类别
                        class_counts = {}
                        for cls in result.boxes.cls:
                            class_id = int(cls)
                            class_name = result.names.get(class_id, f'类别{class_id}')
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        # 显示统计表格
                        st.subheader("检测统计")
                        for class_name, count in class_counts.items():
                            st.write(f"- **{class_name}**: {count}个")
                        
                        # 显示详细数据
                        with st.expander("查看详细数据"):
                            st.write(f"图片尺寸: {result.orig_shape}")
                            st.write(f"推理时间: {results[0].speed['inference']:.2f}ms")
                    else:
                        st.warning("未检测到任何目标")

# 添加示例图片
with st.expander("查看示例图片"):
    cols = st.columns(3)
    
    test_dir = "runs\detect\predict"
    example_images = []  # 初始化列表
    
    if os.path.exists(test_dir):
        # 获取测试集中的图片文件
        all_files = os.listdir(test_dir)
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        test_images = image_files[:6]  # 取前6张图片文件
        example_images = [os.path.join(test_dir, img) for img in test_images]  # 使用正确的变量名
    
    # 如果没有找到图片，显示提示
    if not example_images:
        st.info("测试集目录中没有找到图片文件。")
    
    for idx, col in enumerate(cols):
        if idx < len(example_images):
            with col:
                # 检查图片是否存在
                if os.path.exists(example_images[idx]):
                    # 显示图片文件名（不包含路径）
                    img_name = os.path.basename(example_images[idx])
                    st.caption(f"示例 {idx+1}: {img_name}")
                    
                    # 显示缩略图
                    st.image(example_images[idx], use_container_width=True)
                
                else:
                    st.warning(f"图片不存在: {example_images[idx]}")
st.markdown("---")
st.markdown("**毕业设计项目** | 基于YOLO的工厂安全帽检测系统")