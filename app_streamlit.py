# app_streamlit.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# 页面设置
st.set_page_config(
    page_title="工厂安全检测系统",
    page_icon="🏭",
    layout="wide"
)

# 标题
st.title("🏭 工厂安全检测系统")
st.markdown("上传工厂场景图片，自动检测工人安全防护装备佩戴情况")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    
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
    st.header("📋 检测类别")
    st.markdown("""
    - **helmet**: 佩戴安全帽
    - **no-helmet**: 未佩戴安全帽
    """)

# 主界面
col1, col2 = st.columns(2)

with col1:
    st.header("📤 上传图片")
    
    # 图片上传
    uploaded_file = st.file_uploader(
        "选择一张工厂场景图片",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        # 显示原图
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_column_width=True)
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 检测按钮
        if st.button("🚀 开始检测", type="primary"):
            with st.spinner("检测中..."):
                # 加载模型
                model_path = 'runs/detect/safety_detection_v1/weights/best.pt'
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
                    st.header("📊 检测结果")
                    
                    # 显示结果图片
                    st.image(result_img, caption="检测结果", use_column_width=True)
                    
                    # 显示统计信息
                    if result.boxes is not None:
                        num_detections = len(result.boxes)
                        st.success(f"✅ 检测到 {num_detections} 个目标")
                        
                        # 统计各类别
                        class_counts = {}
                        for cls in result.boxes.cls:
                            class_id = int(cls)
                            class_name = result.names.get(class_id, f'类别{class_id}')
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        # 显示统计表格
                        st.subheader("📈 检测统计")
                        for class_name, count in class_counts.items():
                            st.write(f"- **{class_name}**: {count}个")
                        
                        # 显示详细数据
                        with st.expander("查看详细数据"):
                            st.write(f"图片尺寸: {result.orig_shape}")
                            st.write(f"推理时间: {results[0].speed['inference']:.2f}ms")
                    else:
                        st.warning("⚠️ 未检测到任何目标")

# 添加示例图片
with st.expander("🖼️ 查看示例图片"):
    st.markdown("""
    你可以使用以下类型的图片进行测试：
    1. 工厂/工地场景
    2. 工人密集区域
    3. 不同光照条件
    4. 有遮挡的情况
    """)

# 页脚
st.markdown("---")
st.markdown("**毕业设计项目** | 基于YOLOv8的工厂安全检测系统")