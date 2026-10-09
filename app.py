import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="DWSIM Distillation Surrogate Model",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS for clean industrial styling and zero ghost elements
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        border-left: 5px solid #FF4B4B;
        padding-left: 15px;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1rem;
        color: #8A92A6;
        margin-bottom: 25px;
        padding-left: 20px;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 15px;
        border-bottom: 1px solid #2C303E;
        padding-bottom: 5px;
    }
    .metric-card {
        background-color: #1E212A;
        border: 1px solid #2C303E;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        height: 100%;
        margin-top: 10px;
    }
    .inner-box {
        background-color: #262B36;
        border: 1px solid #3A4050;
        padding: 12px 15px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .product-desc {
        font-size: 0.85rem;
        color: #8A92A6;
        margin-bottom: 20px;
        font-style: italic;
    }
    .stream-header-box {
        background-color: #262B36;
        border: 1px solid #3A4050;
        padding: 10px 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        color: #60A5FA;
        margin-bottom: 15px;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    .diag-box {
        background: linear-gradient(135deg, #1E212A 0%, #262B36 100%);
        border: 1px solid #3A4050;
        padding: 14px;
        border-radius: 10px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Load Model and Scaler Safely
@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "best_surrogate_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# 3. Simple, Clean Header (No box, no emojis, clean title)
st.markdown('<p class="main-header">Benzene-Toluene Distillation Surrogate Model</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time thermodynamic prediction engine powered by DWSIM & Optimized Random Forest Machine Learning</p>', unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("⚠️ Model or Scaler artifacts not found! Please run `python src/train.py` first.")
else:
    # Sidebar Controls
    st.sidebar.markdown("### Feed Control Panel")
    st.sidebar.markdown("Adjust feed parameters to simulate real-time column responses.")
    
    feed_t = st.sidebar.slider("Feed Temperature (K)", 330.0, 380.0, 350.0, 0.5)
    feed_p = 101325.0  # Constant pressure in Pa
    feed_flow = st.sidebar.slider("Feed Mass Flow Rate (kg/s)", 0.5, 3.0, 1.5, 0.1)
    feed_bz = st.sidebar.slider("Feed Benzene Mole Fraction", 0.2, 0.8, 0.5, 0.01)
    
    feed_tol = 1.0 - feed_bz
    st.sidebar.markdown(f"**Calculated Toluene Fraction:** `{feed_tol:.2f}`")
    
    # Model Diagnostics Expander in Sidebar
    with st.sidebar.expander("🛠️ Model Diagnostics & Specs", expanded=False):
        st.markdown(f"""
        <div class="diag-box">
            <span style="color: #60A5FA; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;">ALGORITHM</span><br>
            <span style="color: #FFFFFF; font-size: 0.9rem; font-weight: 600;">Random Forest Regressor</span><br><br>
            <span style="color: #60A5FA; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;">OPTIMIZATION</span><br>
            <span style="color: #FFFFFF; font-size: 0.9rem; font-weight: 600;">RandomizedSearchCV (3-Fold)</span><br><br>
            <span style="color: #60A5FA; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;">ESTIMATORS / DEPTH</span><br>
            <span style="color: #FFFFFF; font-size: 0.9rem; font-weight: 600;">N = {getattr(model, 'n_estimators', 200)} | Depth = {getattr(model, 'max_depth', 'None')}</span><br><br>
            <span style="color: #34D399; font-size: 0.8rem; font-weight: 600;">● Status: Production Ready</span>
        </div>
        """, unsafe_allow_html=True)

    # Main Dashboard layout split into two columns
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown('<p class="section-title">Selected Inputs</p>', unsafe_allow_html=True)
        st.info(f"""
        * **Temperature:** {feed_t} K
        * **Pressure:** {feed_p} Pa
        * **Mass Flow:** {feed_flow} kg/s
        * **Benzene (Feed):** {feed_bz:.2f}
        * **Toluene (Feed):** {feed_tol:.2f}
        """)
        
        run_btn = st.button("Simulate & Predict", type="primary", use_container_width=True)

    with col_right:
        if run_btn:
            # Format input dataframe with explicit feature names
            input_df = pd.DataFrame([[feed_t, feed_p, feed_flow, feed_bz, feed_tol]], 
                                    columns=["Feed_T", "Feed_P", "Feed_Flow", "Feed_Benzene", "Feed_Toluene"])
            features_scaled = scaler.transform(input_df)
            
            # Predict outputs
            prediction = model.predict(features_scaled)[0]
            
            dist_t, dist_p_out, dist_flow, dist_bz, dist_tol = prediction[0], prediction[1], prediction[2], prediction[3], prediction[4]
            bot_t, bot_p_out, bot_flow, bot_bz, bot_tol = prediction[5], prediction[6], prediction[7], prediction[8], prediction[9]

            st.success("✨ Prediction computed successfully via Optimized ML Surrogate Model!")
            
            # Results Cards Columns
            res1, res2 = st.columns(2)
            
            with res1:
                
                st.markdown('<div class="stream-header-box">OVERHEAD STREAM PROFILE</div>', unsafe_allow_html=True)
                st.markdown("#### Distillate - Overhead Product")
                st.markdown('<p class="product-desc">The purified light product collected at the top of the column, highly enriched in volatile Benzene.</p>', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="inner-box">
                    <span style="color: #8A92A6; font-size: 0.85rem;">Overhead Temperature</span><br>
                    <span style="font-size: 1.5rem; font-weight: 700; color: #FFFFFF;">{dist_t:.2f} K</span>
                </div>
                <div class="inner-box">
                    <span style="color: #8A92A6; font-size: 0.85rem;">Distillate Flow Rate</span><br>
                    <span style="font-size: 1.5rem; font-weight: 700; color: #FFFFFF;">{dist_flow:.4f} kg/s</span>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown("**Composition Breakdown:**")
                st.progress(float(dist_bz), text=f"Benzene: {dist_bz*100:.1f}%")
                st.progress(float(dist_tol), text=f"Toluene: {dist_tol*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with res2:
                
                st.markdown('<div class="stream-header-box">BOTTOMS STREAM PROFILE</div>', unsafe_allow_html=True)
                st.markdown("#### Bottoms Product")
                st.markdown('<p class="product-desc">The heavy liquid product recovered from the column base, rich in higher-boiling Toluene.</p>', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="inner-box">
                    <span style="color: #8A92A6; font-size: 0.85rem;">Bottoms Temperature</span><br>
                    <span style="font-size: 1.5rem; font-weight: 700; color: #FFFFFF;">{bot_t:.2f} K</span>
                </div>
                <div class="inner-box">
                    <span style="color: #8A92A6; font-size: 0.85rem;">Bottoms Flow Rate</span><br>
                    <span style="font-size: 1.5rem; font-weight: 700; color: #FFFFFF;">{bot_flow:.4f} kg/s</span>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown("**Composition Breakdown:**")
                st.progress(float(bot_bz), text=f"Benzene: {bot_bz*100:.1f}%")
                st.progress(float(bot_tol), text=f"Toluene: {bot_tol*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            # --- CSV Export Feature ---
            st.markdown("<br>", unsafe_allow_html=True)
            report_df = pd.DataFrame({
                "Date_Time": [datetime.now().strftime("%d-%b-%Y %H:%M")],
                "Feed_Temp_K": [round(feed_t, 2)],
                "Feed_Flow_kg_s": [round(feed_flow, 2)],
                "Feed_Benzene": [round(feed_bz, 3)],
                "Feed_Toluene": [round(feed_tol, 3)],
                "Dist_Temp_K": [round(dist_t, 2)],
                "Dist_Flow_kg_s": [round(dist_flow, 4)],
                "Dist_Benzene": [round(dist_bz, 4)],
                "Dist_Toluene": [round(dist_tol, 4)],
                "Bottom_Temp_K": [round(bot_t, 2)],
                "Bottom_Flow_kg_s": [round(bot_flow, 4)],
                "Bottom_Benzene": [round(bot_bz, 4)],
                "Bottom_Toluene": [round(bot_tol, 4)]
            })
            
            csv_data = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Simulation Report (CSV)",
                data=csv_data,
                file_name=f"dwsim_surrogate_report_{datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:
            st.markdown("""
            ### Welcome to the Dashboard
            Use the **Feed Control Panel** on the left to configure your operating conditions, then click **Simulate & Predict** to view real-time distillate and bottoms responses.
            """)