import React, { useState, useRef, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { Play, Square, RotateCcw, Download, Trash2, Clock, TrendingUp } from 'lucide-react';

export default function WellnessEmotionDetection() {
  const [isRecording, setIsRecording] = useState(false);
  const [emotionTimeline, setEmotionTimeline] = useState([]);
  const [emotionStats, setEmotionStats] = useState({
    'Angry': 0,
    'Disgust': 0,
    'Fear': 0,
    'Happy': 0,
    'Neutral': 0,
    'Sad': 0,
    'Surprise': 0
  });
  const [currentEmotion, setCurrentEmotion] = useState('Neutral');
  const [sessionTime, setSessionTime] = useState(0);
  const [sessionStartTime, setSessionStartTime] = useState(null);
  const videoRef = useRef(null);
  const timerRef = useRef(null);
  const emotionIntervalRef = useRef(null);

  const COLORS = {
    'Angry': '#FF4444',
    'Disgust': '#FF8C00',
    'Fear': '#8B008B',
    'Happy': '#FFD700',
    'Neutral': '#808080',
    'Sad': '#4169E1',
    'Surprise': '#FF1493'
  };

  const emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];

  useEffect(() => {
    if (isRecording) {
      const startCamera = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        } catch (err) {
          console.error('Error accessing camera:', err);
          alert('Please allow camera access to use this feature');
        }
      };
      startCamera();
      setSessionStartTime(new Date());

      timerRef.current = setInterval(() => {
        setSessionTime(t => t + 1);
      }, 1000);

      // Simulate emotion detection every 3 seconds
      emotionIntervalRef.current = setInterval(() => {
        const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];
        
        setCurrentEmotion(randomEmotion);
        setEmotionStats(prev => ({
          ...prev,
          [randomEmotion]: prev[randomEmotion] + 1
        }));

        // Add to timeline with timestamp
        setEmotionTimeline(prev => [...prev, {
          time: formatTimelineTime(sessionTime),
          emotion: randomEmotion,
          timestamp: new Date().toLocaleTimeString()
        }]);
      }, 3000);

      return () => {
        clearInterval(emotionIntervalRef.current);
        if (videoRef.current?.srcObject) {
          videoRef.current.srcObject.getTracks().forEach(track => track.stop());
        }
      };
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      if (emotionIntervalRef.current) clearInterval(emotionIntervalRef.current);
    }
  }, [isRecording, sessionTime]);

  const handleReset = () => {
    setEmotionStats({
      'Angry': 0,
      'Disgust': 0,
      'Fear': 0,
      'Happy': 0,
      'Neutral': 0,
      'Sad': 0,
      'Surprise': 0
    });
    setEmotionTimeline([]);
    setSessionTime(0);
    setCurrentEmotion('Neutral');
    setSessionStartTime(null);
  };

  const handleDownload = () => {
    const report = {
      sessionDate: sessionStartTime ? sessionStartTime.toLocaleString() : 'N/A',
      sessionDuration: formatTime(sessionTime),
      emotionStats,
      emotionTimeline,
      summary: generateSummary()
    };

    const dataStr = JSON.stringify(report, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `emotion-report-${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const generateSummary = () => {
    const total = Object.values(emotionStats).reduce((a, b) => a + b, 0);
    if (total === 0) return 'No data recorded';
    
    const dominant = Object.entries(emotionStats).reduce((a, b) => 
      b[1] > a[1] ? b : a
    )[0];
    
    const percentage = ((emotionStats[dominant] / total) * 100).toFixed(1);
    return `Dominant emotion: ${dominant} (${percentage}%)`;
  };

  const formatTime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const formatTimelineTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const chartData = emotions.map(emotion => ({
    name: emotion,
    count: emotionStats[emotion]
  }));

  const totalDetections = Object.values(emotionStats).reduce((a, b) => a + b, 0);
  const dominantEmotion = totalDetections > 0 
    ? Object.entries(emotionStats).reduce((a, b) => b[1] > a[1] ? b : a)[0]
    : 'None';

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-2">🧘 Wellness Emotion Tracking</h1>
          <p className="text-blue-200 text-lg">Real-time Emotional Wellness Monitoring & Analysis</p>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Video Feed */}
          <div className="lg:col-span-2">
            <div className="bg-black rounded-2xl overflow-hidden shadow-2xl border-4 border-blue-500">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="w-full aspect-video object-cover"
              />
            </div>
          </div>

          {/* Right Panel */}
          <div className="space-y-4">
            {/* Current Emotion */}
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-blue-400">
              <h3 className="text-blue-200 text-sm font-semibold mb-4">CURRENT EMOTION</h3>
              <div className="text-6xl font-bold mb-6" style={{ color: COLORS[currentEmotion] }}>
                {currentEmotion}
              </div>
            </div>

            {/* Session Info */}
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-blue-400">
              <h3 className="text-blue-200 text-sm font-semibold mb-4">SESSION INFO</h3>
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Clock size={20} className="text-blue-300" />
                  <div>
                    <p className="text-blue-200 text-xs">Duration</p>
                    <p className="text-2xl font-bold text-white">{formatTime(sessionTime)}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <TrendingUp size={20} className="text-blue-300" />
                  <div>
                    <p className="text-blue-200 text-xs">Total Detections</p>
                    <p className="text-2xl font-bold text-white">{totalDetections}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Dominant Emotion */}
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-blue-400">
              <h3 className="text-blue-200 text-sm font-semibold mb-4">DOMINANT EMOTION</h3>
              <div className="flex items-center gap-3">
                <div
                  className="w-12 h-12 rounded-full"
                  style={{ backgroundColor: COLORS[dominantEmotion] }}
                />
                <div>
                  <p className="text-white text-xl font-bold">{dominantEmotion}</p>
                  <p className="text-blue-200 text-sm">
                    {totalDetections > 0 ? `${((emotionStats[dominantEmotion] / totalDetections) * 100).toFixed(1)}%` : '0%'}
                  </p>
                </div>
              </div>
            </div>

            {/* Controls */}
            <div className="flex gap-2 flex-col">
              <button
                onClick={() => setIsRecording(!isRecording)}
                className={`py-3 rounded-xl font-bold text-white flex items-center justify-center gap-2 transition-all transform hover:scale-105 ${
                  isRecording
                    ? 'bg-red-600 hover:bg-red-700'
                    : 'bg-green-600 hover:bg-green-700'
                }`}
              >
                {isRecording ? (
                  <>
                    <Square size={20} />
                    Stop Session
                  </>
                ) : (
                  <>
                    <Play size={20} />
                    Start Session
                  </>
                )}
              </button>
              <div className="flex gap-2">
                <button
                  onClick={handleDownload}
                  disabled={totalDetections === 0}
                  className="flex-1 py-2 rounded-xl font-bold text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:opacity-50 flex items-center justify-center gap-2 transition-all"
                >
                  <Download size={18} />
                  Report
                </button>
                <button
                  onClick={handleReset}
                  className="flex-1 py-2 rounded-xl font-bold text-white bg-gray-600 hover:bg-gray-700 flex items-center justify-center gap-2 transition-all"
                >
                  <RotateCcw size={18} />
                  Reset
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Analytics Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Bar Chart - Emotion Distribution */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-blue-400">
            <h3 className="text-white text-xl font-bold mb-4">📊 Emotion Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis dataKey="name" stroke="#aaa" />
                <YAxis stroke="#aaa" />
                <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #666' }} />
                <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                  {emotions.map(emotion => (
                    <Cell key={emotion} fill={COLORS[emotion]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Pie Chart - Emotion Breakdown */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-blue-400">
            <h3 className="text-white text-xl font-bold mb-4">🥧 Emotion Breakdown</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {emotions.map((emotion) => (
                    <Cell key={emotion} fill={COLORS[emotion]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #666' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Timeline Graph */}
        {emotionTimeline.length > 0 && (
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-blue-400 mb-6">
            <h3 className="text-white text-xl font-bold mb-4">📈 Emotion Timeline</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={emotionTimeline.map((item, idx) => ({
                time: item.time,
                value: emotions.indexOf(item.emotion) + 1,
                emotion: item.emotion
              }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis dataKey="time" stroke="#aaa" />
                <YAxis stroke="#aaa" domain={[0, 7]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #666' }}
                  formatter={(value) => emotions[value - 1]}
                />
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#3b82f6" 
                  dot={{ fill: '#3b82f6', r: 4 }}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Emotion History Table */}
        {emotionTimeline.length > 0 && (
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-blue-400 mb-6">
            <h3 className="text-white text-xl font-bold mb-4">📋 Emotion History</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-blue-400">
                    <th className="text-left py-2 px-4 text-blue-200">Time</th>
                    <th className="text-left py-2 px-4 text-blue-200">Emotion</th>
                    <th className="text-left py-2 px-4 text-blue-200">Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {emotionTimeline.slice(-20).map((item, idx) => (
                    <tr key={idx} className="border-b border-blue-900 hover:bg-white/5">
                      <td className="py-2 px-4 text-white">{item.time}</td>
                      <td className="py-2 px-4">
                        <span 
                          className="px-3 py-1 rounded-full text-white font-semibold text-xs"
                          style={{ backgroundColor: COLORS[item.emotion] }}
                        >
                          {item.emotion}
                        </span>
                      </td>
                      <td className="py-2 px-4 text-blue-200">{item.timestamp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Footer Info */}
        <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-blue-400">
          <h3 className="text-white font-bold mb-3">💡 Wellness Center Usage</h3>
          <p className="text-blue-200 text-sm">
            This system tracks emotional states to provide insights into your wellness journey. All data is stored locally and can be exported as reports for personal wellness tracking or professional consultation.
          </p>
        </div>
      </div>
    </div>
  );
}