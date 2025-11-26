import streamlit as st
import sys
import os
from pathlib import Path
import numpy as np

# 嘗試導入必要的庫，提供清晰的錯誤信息
try:
    import joblib
    from sklearn.datasets import load_iris
except ImportError as e:
    st.error(f"❌ 導入錯誤: {e}")
    st.info("請確保 requirements.txt 中包含所有必要的套件")
    st.stop()

# 設置頁面配置
st.set_page_config(
    page_title="Iris 分類器",
    page_icon="🌷",
    layout="wide"
)

# 應用標題
st.title("🌷 Iris 鳶尾花分類器")
st.markdown("使用機器學習模型預測鳶尾花種類")

# 改進的模型加載函數
@st.cache_resource
def load_models():
    """加載模型文件，使用緩存提高性能"""
    try:
        # 嘗試多個可能的路徑
        possible_paths = [
            Path('.'),  # 當前目錄
            Path('./models'),  # models 子目錄
        ]
        
        model, scaler = None, None
        
        for base_path in possible_paths:
            model_path = base_path / 'model.joblib'
            scaler_path = base_path / 'scaler.joblib'
            
            if model_path.exists() and scaler_path.exists():
                try:
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    st.success(f"✅ 從 {base_path} 加載模型成功")
                    break
                except Exception as e:
                    st.error(f"❌ 加載模型失敗: {e}")
                    continue
        
        if model is None or scaler is None:
            st.error("🚫 無法找到或加載模型文件")
            st.info("""
            請確保以下文件存在：
            - `model.joblib`
            - `scaler.joblib`
            
            這些文件應該在專案的根目錄或 models 文件夾中。
            """)
        
        return model, scaler
        
    except Exception as e:
        st.error(f"❌ 加載模型時發生錯誤: {e}")
        return None, None

# 加載模型
model, scaler = load_models()

# 主應用界面
if model is not None and scaler is not None:
    st.success("🎉 模型加載成功！請輸入特徵值進行預測")
    
    # 創建輸入列
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌿 花萼特徵")
        sepal_length = st.slider(
            "花萼長度 (cm)", 
            min_value=4.0, 
            max_value=8.0, 
            value=5.8, 
            step=0.1,
            help="從花萼底部到頂端的長度"
        )
        sepal_width = st.slider(
            "花萼寬度 (cm)", 
            min_value=2.0, 
            max_value=4.5, 
            value=3.0, 
            step=0.1,
            help="花萼最寬處的寬度"
        )
    
    with col2:
        st.subheader("🌸 花瓣特徵")
        petal_length = st.slider(
            "花瓣長度 (cm)", 
            min_value=1.0, 
            max_value=7.0, 
            value=4.0, 
            step=0.1,
            help="從花瓣底部到頂端的長度"
        )
        petal_width = st.slider(
            "花瓣寬度 (cm)", 
            min_value=0.1, 
            max_value=2.5, 
            value=1.2, 
            step=0.1,
            help="花瓣最寬處的寬度"
        )
    
    # 顯示當前輸入值
    st.subheader("📊 當前輸入值")
    input_data = {
        "花萼長度": f"{sepal_length} cm",
        "花萼寬度": f"{sepal_width} cm", 
        "花瓣長度": f"{petal_length} cm",
        "花瓣寬度": f"{petal_width} cm"
    }
    
    st.json(input_data)
    
    # 預測按鈕
    if st.button("🔮 開始預測", type="primary", use_container_width=True):
        with st.spinner("正在進行預測..."):
            try:
                # 準備輸入數據
                input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                
                # 特徵縮放
                input_scaled = scaler.transform(input_features)
                
                # 進行預測
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)
                
                # 顯示結果
                species = ['Setosa', 'Versicolor', 'Virginica']
                result = species[prediction[0]]
                
                st.success("## 🎯 預測完成！")
                
                # 顯示主要結果
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.metric(
                        label="**預測種類**", 
                        value=result,
                        delta="高置信度" if np.max(prediction_proba[0]) > 0.8 else "中等置信度"
                    )
                    
                with result_col2:
                    confidence = np.max(prediction_proba[0])
                    st.metric(
                        label="**置信度**", 
                        value=f"{confidence:.1%}"
                    )
                
                # 詳細概率分佈
                st.subheader("📈 詳細概率分佈")
                
                for i, (species_name, prob) in enumerate(zip(species, prediction_proba[0])):
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        st.write(f"**{species_name}**")
                    
                    with col2:
                        st.progress(float(prob))
                    
                    with col3:
                        st.write(f"**{prob:.1%}**")
                
                # 特徵解釋
                st.subheader("💡 關於 Iris 數據集")
                st.info("""
                **鳶尾花種類說明：**
                - **Setosa**: 最容易識別，花萼較大，花瓣較小
                - **Versicolor**: 中等大小，特徵介於兩者之間  
                - **Virginica**: 花萼較小，花瓣較大
                """)
                
            except Exception as e:
                st.error(f"❌ 預測過程中發生錯誤: {e}")
else:
    st.error("無法啟動應用程式，請檢查模型文件")

# 側邊欄信息
with st.sidebar:
    st.header("ℹ️ 關於此應用")
    st.markdown("""
    這是一個基於機器學習的鳶尾花分類器，使用以下技術：
    
    - 🐍 Python + Streamlit
    - 🤖 Scikit-learn 機器學習
    - 🌐 Render.com 部署
    - 📊 實時預測界面
    """)
    
    st.header("🔧 系統狀態")
    st.write(f"Python 版本: {sys.version.split()[0]}")
    
    # 顯示文件結構
    if st.checkbox("顯示文件結構"):
        st.write("當前目錄文件:")
        for file in Path('.').glob('*'):
            icon = "📄" if file.is_file() else "📁"
            st.write(f"{icon} {file.name}")