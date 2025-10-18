import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import logging
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# Suppress warnings
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("opencv").setLevel(logging.CRITICAL)

# Suppress OpenCV camera warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Page config
st.set_page_config(
    page_title="Wellness Emotion Detection",
    page_icon="🧘",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: rgba(255,255,255,0.1);
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Wellness Emotion Detection & Analytics")
st.markdown("Real-time emotion detection with data visualization and storage")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input("Model Path", "/home/shreyas/Downloads/model(3).h5")
cascade_path = st.sidebar.text_input("Cascade Classifier Path", "/home/shreyas/Downloads/haarcascade_frontalface_default.xml")

# Camera mode selection
st.sidebar.markdown("---")
st.sidebar.header("🎥 Camera Mode")
camera_mode = st.sidebar.radio(
    "Select input method:",
    ["Local Webcam", "Upload Image", "Demo Mode"],
    help="Choose 'Upload Image' or 'Demo Mode' if running on Streamlit Cloud"
)

# Session history storage
HISTORY_DIR = Path("wellness_sessions")
HISTORY_DIR.mkdir(exist_ok=True)

# Custom JSON Encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Load cascade classifier
@st.cache_resource
def load_cascade():
    if not os.path.exists(cascade_path):
        st.warning(f"Cascade path not found. Using default cascade...")
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    else:
        cascade = cv2.CascadeClassifier(cascade_path)
    
    if cascade.empty():
        st.error(f"Error loading cascade classifier")
        return None
    return cascade

# Load emotion model
@st.cache_resource
def load_emotion_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        return None
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Check camera availability
def is_camera_available():
    """Check if camera is available on this system"""
    try:
        cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception:
        return False

# Load resources
face_classifier = load_cascade()
classifier = load_emotion_model(model_path)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

if face_classifier is None or classifier is None:
    st.warning("⚠️ Model or cascade classifier not loaded. Some features may be unavailable.")

# Emotion detection function
def detect_emotion(frame):
    """Detect emotions in frame"""
    if face_classifier is None or classifier is None:
        return frame, []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    detections = []
    
    for (x, y, w, h) in faces:
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            prediction = classifier.predict(roi, verbose=0)[0]
            emotion_idx = int(prediction.argmax())
            emotion = emotion_labels[emotion_idx]
            confidence = float(prediction[emotion_idx])
            
            label_position = (x, y - 10)
            cv2.putText(frame, f"{emotion}", label_position,
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            detections.append({
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
    
    return frame, detections

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
    st.session_state.emotion_data = []
    st.session_state.session_start = None
    st.session_state.frame_count = 0

# TABS
tab1, tab2, tab3 = st.tabs(["📹 Live Detection", "📊 View Sessions", "📈 Analytics"])

# ============ TAB 1: LIVE DETECTION ============
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Live Emotion Detection Feed")
        
        col_start, col_stop, col_reset = st.columns(3)
        
        with col_start:
            if st.button("▶️ Start Recording", use_container_width=True, key="start_btn"):
                if camera_mode == "Local Webcam" and not is_camera_available():
                    st.error("❌ Camera not available. Please select 'Upload Image' or 'Demo Mode' from settings.")
                else:
                    st.session_state.recording = True
                    st.session_state.emotion_data = []
                    st.session_state.session_start = datetime.now()
                    st.session_state.frame_count = 0
                    st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop Recording", use_container_width=True, key="stop_btn"):
                st.session_state.recording = False
                st.rerun()
        
        with col_reset:
            if st.button("🔄 Clear Data", use_container_width=True, key="reset_btn"):
                st.session_state.emotion_data = []
                st.session_state.session_start = None
                st.rerun()
        
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # LOCAL WEBCAM MODE
        if camera_mode == "Local Webcam":
            if st.session_state.recording:
                try:
                    cap = cv2.VideoCapture(0)
                    
                    if cap is None or not cap.isOpened():
                        st.error("❌ Cannot open webcam. Please check if camera is connected.")
                        st.session_state.recording = False
                    else:
                        status_placeholder.success("🟢 Recording... Detecting emotions in real-time")
                        
                        while st.session_state.recording:
                            ret, frame = cap.read()
                            
                            if not ret:
                                st.error("Failed to read from camera")
                                break
                            
                            frame = cv2.flip(frame, 1)
                            frame_with_detections, detections = detect_emotion(frame)
                            
                            for detection in detections:
                                st.session_state.emotion_data.append(detection)
                            
                            frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, use_container_width=True)
                            
                            st.session_state.frame_count += 1
                        
                        cap.release()
                except Exception as e:
                    st.error(f"❌ Camera error: {str(e)}")
                    st.session_state.recording = False
            else:
                status_placeholder.info("⏸️ Ready to record. Click 'Start Recording' to begin.")
        
        # UPLOAD IMAGE MODE
        elif camera_mode == "Upload Image":
            st.info("📤 Upload an image to detect emotions")
            uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if st.button("🔍 Detect Emotions", use_container_width=True, key="detect_btn"):
                    if classifier is not None and face_classifier is not None:
                        frame_with_detections, detections = detect_emotion(frame)
                        
                        st.session_state.emotion_data.extend(detections)
                        st.session_state.session_start = datetime.now()
                        
                        frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, use_container_width=True)
                        
                        status_placeholder.success(f"✅ Detected {len(detections)} face(s)")
                    else:
                        st.error("Model not loaded properly")
        
        # DEMO MODE
        elif camera_mode == "Demo Mode":
            st.info("🎬 Demo mode - uses simulated emotion data")
            
            if st.session_state.recording:
                status_placeholder.success("🟢 Recording... (Demo Mode)")
                
                demo_emotions = ['Happy', 'Neutral', 'Sad', 'Surprise', 'Happy', 'Angry', 'Happy']
                
                for emotion in demo_emotions:
                    if not st.session_state.recording:
                        break
                    
                    fake_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
                    cv2.putText(fake_frame, f"Detected: {emotion}", (150, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    
                    detection = {
                        'emotion': emotion,
                        'confidence': np.random.uniform(0.8, 0.99),
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.emotion_data.append(detection)
                    
                    frame_rgb = cv2.cvtColor(fake_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, use_container_width=True)
                    
                    st.session_state.frame_count += 1
                    import time
                    time.sleep(0.5)
                
                st.session_state.recording = False
                st.rerun()
            else:
                status_placeholder.info("⏸️ Click 'Start Recording' to begin demo")
    
    with col2:
        st.subheader("📊 Live Stats")
        
        if st.session_state.emotion_data:
            total_detections = len(st.session_state.emotion_data)
            emotion_counts = Counter([d['emotion'] for d in st.session_state.emotion_data])
            dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "N/A"
            
            st.metric("📍 Total Detections", total_detections)
            st.metric("🎭 Dominant Emotion", dominant_emotion)
            
            st.markdown("**Emotion Breakdown:**")
            for emotion in emotion_labels:
                count = emotion_counts.get(emotion, 0)
                pct = (count / total_detections * 100) if total_detections > 0 else 0
                st.write(f"• {emotion}: {count} ({pct:.1f}%)")
            
            if st.button("💾 Save Current Session", use_container_width=True, key="save_session"):
                if st.session_state.emotion_data:
                    session_file = HISTORY_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    session_data = {
                        'session_start': st.session_state.session_start.isoformat() if st.session_state.session_start else None,
                        'session_end': datetime.now().isoformat(),
                        'total_detections': total_detections,
                        'emotion_counts': dict(emotion_counts),
                        'detections': st.session_state.emotion_data
                    }
                    with open(session_file, 'w') as f:
                        json.dump(session_data, f, indent=2, cls=NumpyEncoder)
                    st.success(f"✅ Session saved!\n{session_file.name}")
        else:
            st.info("No data yet. Start recording to see stats.")

# ============ TAB 2: VIEW SESSIONS ============
with tab2:
    st.subheader("📋 All Saved Sessions")
    
    session_files = sorted(HISTORY_DIR.glob("session_*.json"), reverse=True)
    
    # Filter valid session files
    valid_sessions = []
    for session_file in session_files:
        try:
            with open(session_file, 'r') as f:
                json.load(f)
            valid_sessions.append(session_file)
        except json.JSONDecodeError:
            st.warning(f"⚠️ Skipping corrupted file: {session_file.name}")
    
    if valid_sessions:
        selected_session = st.selectbox(
            "Select a session to view",
            valid_sessions,
            format_func=lambda x: x.stem
        )
        
        if selected_session:
            with open(selected_session, 'r') as f:
                session_data = json.load(f)
            
            detections = session_data.get('detections', [])
            emotion_counts = session_data.get('emotion_counts', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Detections", len(detections))
            
            with col2:
                if emotion_counts:
                    top_emotion = max(emotion_counts, key=emotion_counts.get)
                    st.metric("🎭 Top Emotion", top_emotion)
                else:
                    st.metric("🎭 Top Emotion", "N/A")
            
            with col3:
                st.metric("📅 Session Date", selected_session.stem.split('_')[1][:8])
            
            with col4:
                st.metric("⏱️ Total Records", len(detections))
            
            # Emotion breakdown
            st.markdown("**Emotion Distribution:**")
            emotion_df = pd.DataFrame([
                {'Emotion': emotion, 'Count': emotion_counts.get(emotion, 0)}
                for emotion in emotion_labels
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_bar = px.bar(
                    emotion_df,
                    x='Emotion',
                    y='Count',
                    color='Emotion',
                    color_discrete_map={
                        'Angry': '#FF4444',
                        'Disgust': '#FF8C00',
                        'Fear': '#8B008B',
                        'Happy': '#FFD700',
                        'Neutral': '#808080',
                        'Sad': '#4169E1',
                        'Surprise': '#FF1493'
                    },
                    title="Emotion Frequency"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                fig_pie = px.pie(
                    emotion_df[emotion_df['Count'] > 0],
                    values='Count',
                    names='Emotion',
                    color='Emotion',
                    color_discrete_map={
                        'Angry': '#FF4444',
                        'Disgust': '#FF8C00',
                        'Fear': '#8B008B',
                        'Happy': '#FFD700',
                        'Neutral': '#808080',
                        'Sad': '#4169E1',
                        'Surprise': '#FF1493'
                    },
                    title="Emotion Breakdown"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Timeline
            if detections:
                st.markdown("**Emotion Timeline:**")
                
                timeline_data = []
                for idx, detection in enumerate(detections):
                    timeline_data.append({
                        'Index': idx,
                        'Emotion': detection['emotion'],
                        'Confidence': detection['confidence'],
                        'Time': detection['timestamp']
                    })
                
                timeline_df = pd.DataFrame(timeline_data)
                
                emotion_to_num = {e: i+1 for i, e in enumerate(emotion_labels)}
                timeline_df['EmotionNum'] = timeline_df['Emotion'].map(emotion_to_num)
                
                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Scatter(
                    x=timeline_df['Index'],
                    y=timeline_df['EmotionNum'],
                    mode='lines+markers',
                    name='Emotion',
                    hovertemplate='<b>%{customdata}</b><extra></extra>',
                    customdata=timeline_df['Emotion'],
                    line=dict(color='#667eea', width=2),
                    marker=dict(size=6)
                ))
                
                fig_timeline.update_layout(
                    title="Emotion Progression",
                    xaxis_title="Detection #",
                    yaxis_title="Emotion",
                    hovermode='x unified',
                    template='plotly_dark',
                    height=400
                )
                fig_timeline.update_yaxes(
                    tickvals=list(emotion_to_num.values()),
                    ticktext=list(emotion_to_num.keys())
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Session info
            st.markdown("**Session Information:**")
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.write(f"**Start Time:** {session_data.get('session_start', 'N/A')}")
                st.write(f"**End Time:** {session_data.get('session_end', 'N/A')}")
            
            with info_col2:
                st.write(f"**File:** {selected_session.name}")
    else:
        st.info("No saved sessions yet. Start recording to create one!")

# ============ TAB 3: ANALYTICS ============
with tab3:
    st.subheader("📈 Overall Analytics")
    
    session_files = sorted(HISTORY_DIR.glob("session_*.json"), reverse=True)
    
    # Filter valid session files
    valid_sessions = []
    for session_file in session_files:
        try:
            with open(session_file, 'r') as f:
                json.load(f)
            valid_sessions.append(session_file)
        except json.JSONDecodeError:
            continue
    
    if valid_sessions:
        all_emotions = []
        all_sessions_data = []
        
        for session_file in valid_sessions:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            emotion_counts = session_data.get('emotion_counts', {})
            total_detections = sum(emotion_counts.values())
            
            all_sessions_data.append({
                'Session': session_file.stem,
                'Total Detections': total_detections,
                'Top Emotion': max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'N/A'
            })
            
            for emotion, count in emotion_counts.items():
                all_emotions.extend([emotion] * count)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Sessions", len(valid_sessions))
        
        with col2:
            st.metric("📍 Total Detections", len(all_emotions))
        
        with col3:
            if all_emotions:
                counter = Counter(all_emotions)
                top_emotion = counter.most_common(1)[0][0]
                st.metric("🎭 Most Common", top_emotion)
            else:
                st.metric("🎭 Most Common", "N/A")
        
        with col4:
            st.metric("📈 Avg/Session", len(all_emotions) // len(valid_sessions) if valid_sessions else 0)
        
        st.markdown("**Overall Emotion Distribution:**")
        
        emotion_counter = Counter(all_emotions)
        overall_df = pd.DataFrame([
            {'Emotion': emotion, 'Count': emotion_counter.get(emotion, 0)}
            for emotion in emotion_labels
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_overall_bar = px.bar(
                overall_df,
                x='Emotion',
                y='Count',
                color='Emotion',
                color_discrete_map={
                    'Angry': '#FF4444',
                    'Disgust': '#FF8C00',
                    'Fear': '#8B008B',
                    'Happy': '#FFD700',
                    'Neutral': '#808080',
                    'Sad': '#4169E1',
                    'Surprise': '#FF1493'
                },
                title="Overall Emotion Frequency"
            )
            st.plotly_chart(fig_overall_bar, use_container_width=True)
        
        with col2:
            fig_overall_pie = px.pie(
                overall_df[overall_df['Count'] > 0],
                values='Count',
                names='Emotion',
                color='Emotion',
                color_discrete_map={
                    'Angry': '#FF4444',
                    'Disgust': '#FF8C00',
                    'Fear': '#8B008B',
                    'Happy': '#FFD700',
                    'Neutral': '#808080',
                    'Sad': '#4169E1',
                    'Surprise': '#FF1493'
                },
                title="Overall Emotion Breakdown"
            )
            st.plotly_chart(fig_overall_pie, use_container_width=True)
        
        st.markdown("**Session History:**")
        sessions_df = pd.DataFrame(all_sessions_data)
        st.dataframe(sessions_df, use_container_width=True)
    else:
        st.info("No sessions yet. Start recording to see analytics!")

# Footer
st.markdown("---")
st.markdown("""
**Wellness Emotion Detection System**
- Real-time emotion detection with Haar Cascade & CNN
- Multiple input modes: Local Webcam, Upload Image, Demo Mode
- Automatic data storage in JSON format
- Interactive visualizations with Plotly
- 7 Emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
""")
