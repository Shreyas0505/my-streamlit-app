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

# --- WebRTC Imports ---
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Suppress warnings
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- WebRTC Configuration ---
# Use Google's public STUN server for ICE negotiation
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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
        color: white; /* Ensure text is visible on gradient */
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Wellness Emotion Detection & Analytics")
st.markdown("Real-time emotion detection with data visualization and storage, powered by WebRTC.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")
model_path = st.sidebar.text_input("Model Path", "model(3).h5")
cascade_path = st.sidebar.text_input("Cascade Classifier Path", "haarcascade_frontalface_default.xml")

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
def load_cascade(path):
    cascade = cv2.CascadeClassifier(path)
    return cascade

# Load emotion model
@st.cache_resource
def load_emotion_model(path):
    try:
        model = load_model(path)
        return model
    except Exception as e:
        return None

# Load resources
face_classifier = load_cascade(cascade_path)
classifier = load_emotion_model(model_path)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

if face_classifier.empty() or classifier is None:
    if face_classifier.empty():
        st.error(f"❌ Error loading cascade classifier from: {cascade_path}. Check file path.")
    if classifier is None:
        st.error(f"❌ Error loading Keras model from: {model_path}. Check file path.")
    st.stop()

# Emotion detection function
def detect_emotion(frame):
    """Detect emotions in frame. Frame is expected in BGR format."""
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

# --- WebRTC Frame Processing Callback ---
def process_frame(frame: av.VideoFrame):
    """Callback function that receives video frames from WebRTC stream."""
    # Convert incoming AV frame to NumPy array (BGR format for OpenCV)
    img = frame.to_ndarray(format="bgr24")
    
    # Perform emotion detection using existing function
    frame_with_detections, detections = detect_emotion(img)

    # Save detections to session state. This automatically updates the stats column (col2).
    # We only record if the session_start timestamp has been set (i.e., the user clicked the button).
    if st.session_state.get('session_start'):
        for detection in detections:
            st.session_state.emotion_data.append(detection)
    
    # Convert the processed NumPy array back to an AV frame for display
    return av.VideoFrame.from_ndarray(frame_with_detections, format="bgr24")


# Initialize session state (removed 'recording' state as WebRTC handles it)
if 'emotion_data' not in st.session_state:
    st.session_state.emotion_data = []
    st.session_state.session_start = None
    st.session_state.frame_count = 0 # Keeping this for potential future optimizations (e.g., frame skipping)

# TABS
tab1, tab2, tab3 = st.tabs(["📹 Live Detection", "📊 View Sessions", "📈 Analytics"])

# ============ TAB 1: LIVE DETECTION ============
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Live Emotion Detection Feed")
        
        # This button is used to initialize/reset the recording session data
        if st.button("▶️ Start/Reset Session Data", use_container_width=True, key="start_live"):
            st.session_state.emotion_data = []
            st.session_state.session_start = datetime.now()
            st.session_state.frame_count = 0
            st.info("✅ Session data tracking started. The WebRTC stream is now live.")

        st.info("Click the 'Start' button in the stream below and grant camera access to begin detection.")
        
        # WebRTC Streamer component
        webrtc_streamer(
            key="emotion_stream",
            mode=WebRtcMode.SENDRECV, # Send video frames to Python and receive processed frames back
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=process_frame,
            media_stream_constraints={"video": True, "audio": False}, # Only use video
        )
        # Note: The WebRTC component runs continuously once started by the user in the browser.
        # The data logging is controlled by checking if st.session_state.session_start is set in process_frame.

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
                st.write(f"• {emotion}: **{count}** ({pct:.1f}%)")
            
            if st.button("💾 Save Current Session", use_container_width=True, key="save_session"):
                if st.session_state.emotion_data:
                    session_file = HISTORY_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    session_data = {
                        # Use get to handle cases where start might be None
                        'session_start': st.session_state.session_start.isoformat() if st.session_state.session_start else "N/A",
                        'session_end': datetime.now().isoformat(),
                        'total_detections': total_detections,
                        'emotion_counts': dict(emotion_counts),
                        'detections': st.session_state.emotion_data
                    }
                    with open(session_file, 'w') as f:
                        json.dump(session_data, f, indent=2, cls=NumpyEncoder)
                    st.success(f"✅ Session saved as: {session_file.name}")
                    st.session_state.emotion_data = [] # Clear live data after saving
                    st.session_state.session_start = None
                    st.rerun()
                else:
                    st.warning("Cannot save: No detections recorded in the current session.")
        else:
            st.info("No data yet. Start the camera and click the 'Start/Reset Session Data' button to begin tracking.")

# ============ TAB 2: VIEW SESSIONS (Unchanged Logic) ============
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
                # Calculate Duration
                start_time_str = session_data.get('session_start')
                end_time_str = session_data.get('session_end')
                duration_str = "N/A"
                if start_time_str and end_time_str and start_time_str != "N/A":
                    try:
                        start_time = datetime.fromisoformat(start_time_str)
                        end_time = datetime.fromisoformat(end_time_str)
                        duration = end_time - start_time
                        # Format to HH:MM:SS, dropping microseconds
                        duration_str = str(duration).split('.')[0]
                    except ValueError:
                        pass # Keep N/A if datetime parsing fails
                
                st.metric("⏱️ Duration", duration_str)
            
            with col4:
                st.metric("📅 Session Date", selected_session.stem.split('_')[1][:8])
            
            # Emotion breakdown
            st.markdown("---")
            st.markdown("**Emotion Distribution:**")
            emotion_df = pd.DataFrame([
                {'Emotion': emotion, 'Count': emotion_counts.get(emotion, 0)}
                for emotion in emotion_labels
            ])
            
            col_bar, col_pie = st.columns(2)
            
            # Define color map for consistency
            color_map = {
                'Angry': '#FF4444', 'Disgust': '#FF8C00', 'Fear': '#8B008B',
                'Happy': '#FFD700', 'Neutral': '#808080', 'Sad': '#4169E1',
                'Surprise': '#FF1493'
            }
            
            with col_bar:
                fig_bar = px.bar(
                    emotion_df,
                    x='Emotion',
                    y='Count',
                    color='Emotion',
                    color_discrete_map=color_map,
                    title="Emotion Frequency"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col_pie:
                fig_pie = px.pie(
                    emotion_df[emotion_df['Count'] > 0],
                    values='Count',
                    names='Emotion',
                    color='Emotion',
                    color_discrete_map=color_map,
                    title="Emotion Breakdown"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Timeline
            if detections:
                st.markdown("---")
                st.markdown("**Emotion Timeline:** (Note: This uses detection index as a proxy for time)")
                
                # Create timeline data
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
                    title="Emotion Progression Over Session",
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
            st.markdown("---")
            st.markdown("**Session Metadata:**")
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.write(f"**Start Time:** {session_data.get('session_start', 'N/A')}")
                st.write(f"**End Time:** {session_data.get('session_end', 'N/A')}")
            
            with info_col2:
                st.write(f"**File:** {selected_session.name}")
    else:
        st.info("No saved sessions yet. Start the camera and track a session in the 'Live Detection' tab to create one!")

# ============ TAB 3: ANALYTICS (Unchanged Logic) ============
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
        # Aggregate all sessions
        all_emotions = []
        all_sessions_data = []
        
        for session_file in valid_sessions:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            emotion_counts = session_data.get('emotion_counts', {})
            total_detections = sum(emotion_counts.values())
            
            top_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'N/A'
            
            # Clean up session name for display
            session_name = session_file.stem.replace("session_", "").replace("_", " ")
            
            all_sessions_data.append({
                'Session ID': session_name,
                'Total Detections': total_detections,
                'Top Emotion': top_emotion
            })
            
            for emotion, count in emotion_counts.items():
                all_emotions.extend([emotion] * count)
        
        # Summary metrics
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
            st.metric("📈 Avg Detections/Session", len(all_emotions) // len(valid_sessions) if valid_sessions else 0)
        
        # Overall emotion distribution
        st.markdown("---")
        st.markdown("**Overall Emotion Distribution:**")
        
        emotion_counter = Counter(all_emotions)
        overall_df = pd.DataFrame([
            {'Emotion': emotion, 'Count': emotion_counter.get(emotion, 0)}
            for emotion in emotion_labels
        ])
        
        col_bar_overall, col_pie_overall = st.columns(2)
        
        with col_bar_overall:
            fig_overall_bar = px.bar(
                overall_df,
                x='Emotion',
                y='Count',
                color='Emotion',
                color_discrete_map=color_map,
                title="Overall Emotion Frequency"
            )
            st.plotly_chart(fig_overall_bar, use_container_width=True)
        
        with col_pie_overall:
            fig_overall_pie = px.pie(
                overall_df[overall_df['Count'] > 0],
                values='Count',
                names='Emotion',
                color='Emotion',
                color_discrete_map=color_map,
                title="Overall Emotion Breakdown"
            )
            st.plotly_chart(fig_overall_pie, use_container_width=True)
        
        # Sessions table
        st.markdown("---")
        st.markdown("**Session History Table:**")
        sessions_df = pd.DataFrame(all_sessions_data)
        st.dataframe(sessions_df, use_container_width=True)
    else:
        st.info("No sessions yet. Start recording to see analytics!")

# Footer
st.markdown("---")
st.markdown("""
**Wellness Emotion Detection System**
- **Real-time Camera Feed** using WebRTC for robust cross-platform performance.
- Real-time emotion detection with Haar Cascade & CNN.
- Automatic data storage in JSON format.
- Interactive visualizations with Plotly.
- 7 Emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
""")
