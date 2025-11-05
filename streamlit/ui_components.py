"""
UI Components for the Chunkformer Streamlit app
"""

import base64
import json
from typing import Dict, List

from utils import prepare_segments_for_player

import streamlit as st
import streamlit.components.v1 as components


def render_custom_css():
    """Render custom CSS for enhanced UI aesthetics with dark/light mode support"""
    st.markdown(
        """
    <style>
    /* CSS Variables for theme support */
    :root {
        --primary-gradient-start: #667eea;
        --primary-gradient-end: #764ba2;
        --text-primary: #2d3748;
        --text-secondary: #4a5568;
        --text-muted: #718096;
        --bg-primary: #ffffff;
        --bg-secondary: #f7fafc;
        --bg-tertiary: #f0f4ff;
        --border-color: #e2e8f0;
        --shadow-sm: rgba(0, 0, 0, 0.08);
        --shadow-md: rgba(102, 126, 234, 0.15);
        --shadow-lg: rgba(102, 126, 234, 0.4);
        --info-bg: #e0f2fe;
        --info-border: #0284c7;
        --success-bg: #dcfce7;
        --success-border: #16a34a;
        --metric-bg: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Dark mode variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #e2e8f0;
            --text-secondary: #cbd5e0;
            --text-muted: #a0aec0;
            --bg-primary: #1a202c;
            --bg-secondary: #2d3748;
            --bg-tertiary: #2d3748;
            --border-color: #4a5568;
            --shadow-sm: rgba(0, 0, 0, 0.3);
            --shadow-md: rgba(138, 180, 248, 0.2);
            --shadow-lg: rgba(138, 180, 248, 0.5);
            --info-bg: rgba(14, 116, 144, 0.3);
            --info-border: #67e8f9;
            --success-bg: rgba(22, 163, 74, 0.3);
            --success-border: #4ade80;
            --metric-bg: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        }
    }

    /* Main page styling */
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
    }

    /* Title and headers - theme aware */
    h1 {
        background: linear-gradient(
            135deg,
            var(--primary-gradient-start) 0%,
            var(--primary-gradient-end) 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    h2 {
        color: var(--text-primary);
        border-bottom: 3px solid var(--primary-gradient-start);
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }

    h3 {
        color: var(--text-secondary);
        font-weight: 700;
    }

    /* Regular text */
    p, div, span {
        color: var(--text-primary);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2 {
        color: var(--primary-gradient-start);
    }

    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li {
        color: var(--text-primary) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(
            135deg,
            var(--primary-gradient-start) 0%,
            var(--primary-gradient-end) 100%
        );
        color: white !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 15px var(--shadow-lg);
    }

    .stButton > button:hover {
        box-shadow: 0 6px 25px var(--shadow-lg);
        transform: translateY(-2px);
    }

    /* File uploader */
    .stFileUploader {
        border-radius: 12px;
        background-color: var(--bg-tertiary);
        border: 2px dashed var(--border-color);
        padding: 1rem;
    }

    .stFileUploader label {
        color: var(--text-primary) !important;
    }

    /* Metrics styling */
    [data-testid="stMetric"] {
        background: var(--metric-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-gradient-start);
    }

    [data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }

    /* Info and alert styling */
    [data-testid="stAlert"] {
        border-radius: 8px;
        padding: 1rem;
    }

    div[data-baseweb="notification"] {
        border-radius: 8px;
        background-color: var(--info-bg);
        border-left: 4px solid var(--info-border);
    }

    div[data-baseweb="notification"] div {
        color: var(--text-primary) !important;
    }

    /* Success notifications */
    .stSuccess {
        background-color: var(--success-bg) !important;
        border-left: 4px solid var(--success-border) !important;
        color: var(--text-primary) !important;
    }

    /* Container styling */
    .element-container {
        color: var(--text-primary);
    }

    /* Divider enhancement */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(
            to right,
            transparent,
            var(--primary-gradient-start),
            transparent
        );
        margin: 2rem 0 !important;
    }

    /* Selectbox styling */
    .stSelectbox label,
    .stSlider label {
        color: var(--text-primary) !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        background-color: var(--bg-tertiary);
        border-color: var(--border-color);
    }

    /* Caption and small text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-style: italic;
    }

    /* Progress bar */
    [data-testid="stProgressBar"] > div > div {
        background-color: var(--primary-gradient-start);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient-start);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-gradient-end);
    }

    /* Download button text */
    .stDownloadButton label {
        color: var(--text-primary) !important;
    }

    .stDownloadButton button {
        background: linear-gradient(
            135deg,
            var(--primary-gradient-start) 0%,
            var(--primary-gradient-end) 100%
        );
        color: white !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }

    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--shadow-lg);
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--primary-gradient-start) !important;
    }

    /* Markdown text in info boxes */
    .stMarkdown {
        color: var(--text-primary);
    }

    /* Hero section text */
    .hero-text {
        color: var(--text-secondary) !important;
    }

    .hero-subtext {
        color: var(--text-muted) !important;
    }

    /* Step cards */
    .step-card {
        background: var(--bg-tertiary);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid var(--border-color);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .step-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            135deg,
            var(--primary-gradient-start),
            var(--primary-gradient-end)
        );
        opacity: 0;
        transition: opacity 0.4s ease;
        z-index: 0;
    }

    .step-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 32px var(--shadow-md);
        border-color: var(--primary-gradient-start);
    }

    .step-card:hover::before {
        opacity: 0.05;
    }

    .step-card > * {
        position: relative;
        z-index: 1;
    }

    .step-card h3 {
        color: var(--primary-gradient-start) !important;
    }

    .step-card p {
        color: var(--text-primary) !important;
    }

    .step-card .step-subtext {
        color: var(--text-muted) !important;
    }

    /* Feature cards hover effect */
    @keyframes pulse-border {
        0%, 100% {
            border-color: var(--border-color);
        }
        50% {
            border-color: var(--primary-gradient-start);
        }
    }

    /* Smooth animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Add animation to elements */
    .step-card,
    [data-testid="column"] > div {
        animation: fadeInUp 0.6s ease-out;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_synchronized_player(
    video_bytes: bytes, mime_type: str, segments: List[Dict], height: int = 620
) -> None:
    """Render a custom HTML player with transcript synchronized to the video

    Args:
        video_bytes: Raw video file bytes
        mime_type: MIME type of the video
        segments: List of transcript segments
        height: Height of the player component in pixels
    """
    if not video_bytes:
        st.warning("No video data available to render the synchronized player.")
        return

    prepared_segments = prepare_segments_for_player(segments)
    if not prepared_segments:
        st.warning("No segments available to synchronize with the video.")
        return

    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    segments_json = json.dumps(prepared_segments, ensure_ascii=False)

    html_content = f"""
    <style>
        #chunkformer-sync-wrapper {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}
        #chunkformer-sync-wrapper .video-pane {{
            flex: 2 1 380px;
            min-width: 320px;
        }}
        #chunkformer-sync-wrapper .video-pane video {{
            width: 100%;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
            display: block;
        }}
        #chunkformer-sync-wrapper .transcript-pane {{
            flex: 1 1 300px;
            overflow-y: auto;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            background: var(--transcript-bg, rgba(240, 244, 255, 0.5));
            color: var(--transcript-text, #1a202c);
            border: 1px solid var(--transcript-border, rgba(102, 126, 234, 0.2));
            align-self: flex-start;
            box-sizing: border-box;
        }}

        /* Dark mode support */
        @media (prefers-color-scheme: dark) {{
            #chunkformer-sync-wrapper .transcript-pane {{
                --transcript-bg: rgba(30, 35, 45, 0.6);
                --transcript-text: #e2e8f0;
                --transcript-border: rgba(138, 180, 248, 0.3);
            }}
        }}

        #chunkformer-sync-wrapper .transcript-pane::-webkit-scrollbar {{
            width: 8px;
        }}
        #chunkformer-sync-wrapper .transcript-pane::-webkit-scrollbar-track {{
            background: var(--scrollbar-track, rgba(102, 126, 234, 0.1));
            border-radius: 8px;
        }}
        #chunkformer-sync-wrapper .transcript-pane::-webkit-scrollbar-thumb {{
            background: var(--scrollbar-thumb, rgba(102, 126, 234, 0.4));
            border-radius: 8px;
        }}
        #chunkformer-sync-wrapper .transcript-pane::-webkit-scrollbar-thumb:hover {{
            background: var(--scrollbar-thumb-hover, rgba(102, 126, 234, 0.6));
        }}

        @media (prefers-color-scheme: dark) {{
            #chunkformer-sync-wrapper .transcript-pane::-webkit-scrollbar-track {{
                --scrollbar-track: rgba(138, 180, 248, 0.15);
            }}
            #chunkformer-sync-wrapper .transcript-pane::-webkit-scrollbar-thumb {{
                --scrollbar-thumb: rgba(138, 180, 248, 0.4);
            }}
            #chunkformer-sync-wrapper .transcript-pane::-webkit-scrollbar-thumb:hover {{
                --scrollbar-thumb-hover: rgba(138, 180, 248, 0.6);
            }}
        }}

        .transcript-segment {{
            border-radius: 10px;
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.6rem;
            transition: all 0.2s ease;
            cursor: pointer;
            background: var(--segment-bg, rgba(255, 255, 255, 0.4));
            border: 1px solid var(--segment-border, transparent);
        }}
        .transcript-segment:hover {{
            background: var(--segment-hover-bg, rgba(102, 126, 234, 0.12));
            border-color: var(--segment-hover-border, rgba(102, 126, 234, 0.3));
            transform: translateX(4px);
        }}
        .transcript-segment.active {{
            background: var(--segment-active-bg, rgba(102, 126, 234, 0.25));
            border: 2px solid var(--segment-active-border, rgba(102, 126, 234, 0.8));
            transform: translateX(6px);
            box-shadow: 0 2px 8px var(--segment-active-shadow, rgba(102, 126, 234, 0.2));
        }}

        @media (prefers-color-scheme: dark) {{
            .transcript-segment {{
                --segment-bg: rgba(45, 55, 72, 0.4);
                --segment-border: rgba(138, 180, 248, 0.1);
            }}
            .transcript-segment:hover {{
                --segment-hover-bg: rgba(138, 180, 248, 0.15);
                --segment-hover-border: rgba(138, 180, 248, 0.4);
            }}
            .transcript-segment.active {{
                --segment-active-bg: rgba(138, 180, 248, 0.25);
                --segment-active-border: rgba(138, 180, 248, 0.9);
                --segment-active-shadow: rgba(138, 180, 248, 0.3);
            }}
        }}

        .transcript-meta {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            opacity: 0.8;
            margin-bottom: 0.35rem;
            color: var(--meta-text, #4a5568);
        }}
        .transcript-meta .transcript-index {{
            font-weight: 700;
            letter-spacing: 0.05em;
            color: var(--index-color, #667eea);
        }}
        .transcript-meta .transcript-timestamp {{
            font-family: "Roboto Mono", "Courier New", monospace;
            font-size: 0.75rem;
        }}

        @media (prefers-color-scheme: dark) {{
            .transcript-meta {{
                --meta-text: #a0aec0;
            }}
            .transcript-meta .transcript-index {{
                --index-color: #8ab4f8;
            }}
        }}

        .transcript-text {{
            line-height: 1.6;
            font-size: 0.95rem;
            white-space: pre-wrap;
            color: var(--text-color, #2d3748);
        }}

        @media (prefers-color-scheme: dark) {{
            .transcript-text {{
                --text-color: #e2e8f0;
            }}
        }}

        @media (max-width: 900px) {{
            #chunkformer-sync-wrapper {{
                flex-direction: column;
            }}
            #chunkformer-sync-wrapper .transcript-pane {{
                height: auto !important;
                max-height: 400px !important;
                min-height: 280px;
            }}
        }}
    </style>
    <div id="chunkformer-sync-wrapper">
        <div class="video-pane">
            <video id="chunkformer-video" controls preload="metadata">
                <source src="data:{mime_type};base64,{video_b64}" type="{mime_type}" />
                Your browser does not support HTML5 video.
            </video>
        </div>
        <div class="transcript-pane" id="chunkformer-transcript"></div>
    </div>
    <script>
        (function() {{
            const segments = {segments_json};
            const transcriptContainer = document.getElementById('chunkformer-transcript');
            const videoEl = document.getElementById('chunkformer-video');

            if (!segments.length || !transcriptContainer || !videoEl) {{
                return;
            }}

            const formatTime = (seconds) => {{
                if (!Number.isFinite(seconds)) {{
                    return '00:00';
                }}
                const hrs = Math.floor(seconds / 3600);
                const mins = Math.floor((seconds % 3600) / 60);
                const secs = Math.floor(seconds % 60);
                const parts = hrs > 0 ? [hrs, mins, secs] : [mins, secs];
                return parts.map((part) => String(part).padStart(2, '0')).join(':');
            }};

            segments.forEach((segment) => {{
                const segmentEl = document.createElement('div');
                segmentEl.className = 'transcript-segment';
                segmentEl.dataset.index = segment.index;
                segmentEl.dataset.start = segment.start;
                segmentEl.dataset.end = segment.end;

                const metaEl = document.createElement('div');
                metaEl.className = 'transcript-meta';

                const indexEl = document.createElement('span');
                indexEl.className = 'transcript-index';
                indexEl.textContent = `#${{segment.index.toString().padStart(2, '0')}}`;

                const timestampEl = document.createElement('span');
                timestampEl.className = 'transcript-timestamp';
                timestampEl.textContent = (
                    `${{formatTime(segment.start)}} – ` +
                    `${{formatTime(segment.end)}}`
                );

                metaEl.appendChild(indexEl);
                metaEl.appendChild(timestampEl);

                const textEl = document.createElement('div');
                textEl.className = 'transcript-text';
                textEl.textContent = segment.text;

                segmentEl.appendChild(metaEl);
                segmentEl.appendChild(textEl);

                transcriptContainer.appendChild(segmentEl);
            }});

            let activeIndex = null;

            const ensureVisible = (el) => {{
                if (!el) return;
                const containerRect = transcriptContainer.getBoundingClientRect();
                const elRect = el.getBoundingClientRect();
                if (elRect.top < containerRect.top || elRect.bottom > containerRect.bottom) {{
                    el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }}
            }};

            const setActive = (index) => {{
                if (index === activeIndex) return;
                activeIndex = index;
                transcriptContainer
                    .querySelectorAll('.transcript-segment')
                    .forEach((node) => node.classList.remove('active'));
                const activeEl = transcriptContainer.querySelector(
                    `.transcript-segment[data-index="${{index}}"]`
                );
                if (activeEl) {{
                    activeEl.classList.add('active');
                    ensureVisible(activeEl);
                }}
            }};

            const findActiveIndex = (currentTime) => {{
                for (const segment of segments) {{
                    if (currentTime >= segment.start && currentTime <= segment.end) {{
                        return segment.index;
                    }}
                }}
                return null;
            }};

            const syncTranscript = () => {{
                const idx = findActiveIndex(videoEl.currentTime || 0);
                if (idx !== null) {{
                    setActive(idx);
                }}
            }};

            let rafHandle = null;
            const scheduleSync = () => {{
                if (rafHandle) {{
                    cancelAnimationFrame(rafHandle);
                }}
                rafHandle = requestAnimationFrame(syncTranscript);
            }};

            videoEl.addEventListener('timeupdate', scheduleSync);
            videoEl.addEventListener('seeked', syncTranscript);
            videoEl.addEventListener('loadedmetadata', syncTranscript);

            transcriptContainer.addEventListener('click', (event) => {{
                const target = event.target.closest('.transcript-segment');
                if (!target) return;
                const seekTime = parseFloat(target.dataset.start);
                if (!Number.isFinite(seekTime)) return;
                videoEl.currentTime = Math.max(0, seekTime + 0.01);
                videoEl.play();
            }});

            // Sync transcript height with video height
            const syncHeight = () => {{
                const videoHeight = videoEl.offsetHeight;
                if (videoHeight > 0) {{
                    // Set height using border-box model (includes padding and border)
                    transcriptContainer.style.height = `${{videoHeight}}px`;
                    transcriptContainer.style.maxHeight = `${{videoHeight}}px`;
                    transcriptContainer.style.minHeight = `${{videoHeight}}px`;
                }}
            }};

            // Initial height sync
            videoEl.addEventListener('loadedmetadata', () => {{
                syncHeight();
                // Additional sync after metadata loads to ensure accurate height
                setTimeout(syncHeight, 50);
            }});

            // Sync on resize
            window.addEventListener('resize', syncHeight);

            // Also sync on video load (in case loadedmetadata already fired)
            if (videoEl.readyState >= 1) {{
                syncHeight();
            }}

            // Fallback: sync after delays to catch all scenarios
            setTimeout(syncHeight, 100);
            setTimeout(syncHeight, 300);
            setTimeout(syncHeight, 500);
            setTimeout(syncHeight, 1000);

            // initial highlight
            syncTranscript();
        }})();
    </script>
    """

    components.html(html_content, height=height, scrolling=True)


def render_hero_section():
    """Render the hero section of the app"""
    st.markdown(
        """
    <div style="text-align: center; padding: 3rem 1rem 2rem 1rem; position: relative;">
        <div style="display: inline-block; position: relative;">
            <h1 style="margin-bottom: 0.5rem; font-size: 2.5rem;">
                🎬 Chunkformer Video Transcription
            </h1>
            <div style="width: 100%; height: 4px; background: linear-gradient(90deg,
            transparent, var(--primary-gradient-start), var(--primary-gradient-end), transparent);
            border-radius: 2px; margin-bottom: 1rem;"></div>
        </div>
        <p class="hero-text" style="font-size: 1.2rem; margin-bottom: 0.5rem; font-weight: 500;">
            Professional Video Transcription with Synchronized Playback
        </p>
        <p class="hero-subtext" style="
            font-size: 1rem;
            max-width: 700px;
            margin: 0 auto;
            line-height: 1.6;
        ">
            Transform your videos into accurate text with
            AI-powered Vietnamese speech recognition.
            Experience seamless synchronization between
            transcript and playback.
        </p>
        <div style="
            margin-top: 2rem;
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: var(--text-secondary);
            ">
                <span style="font-size: 1.5rem;">⚡</span>
                <span>Fast Processing</span>
            </div>
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: var(--text-secondary);
            ">
                <span style="font-size: 1.5rem;">🎯</span>
                <span>High Accuracy</span>
            </div>
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: var(--text-secondary);
            ">
                <span style="font-size: 1.5rem;">🔒</span>
                <span>Privacy First</span>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_landing_page():
    """Render the landing page when no file is uploaded"""

    # Call-to-action section with enhanced design
    st.markdown(
        """
    <div style='text-align: center; padding: 2.5rem 1rem; background: var(--bg-tertiary);
    border-radius: 16px; margin-bottom: 3rem; border: 2px solid var(--border-color);'>
        <div style='font-size: 3.5rem; margin-bottom: 1rem;'>🎬</div>
        <h2 style='color: var(--text-primary); margin-bottom: 1rem; font-size: 1.8rem;'>
            Ready to Get Started?
        </h2>
        <p style='color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 0.5rem;'>
            Upload your video file above to begin transcription
        </p>
        <p style='color: var(--text-muted); font-size: 0.95rem;'>
            Supports MP4, AVI, MOV, MKV, WebM and more
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # How it works section with enhanced visuals
    st.markdown(
        """
    <h2 style='
        text-align: center;
        color: var(--text-primary);
        margin-bottom: 2rem;
        font-size: 1.8rem;
    '>
        🚀 How It Works
    </h2>
    """,
        unsafe_allow_html=True,
    )

    step_cols = st.columns(4)

    steps = [
        {
            "icon": "📤",
            "number": "1",
            "title": "Upload",
            "description": "Choose your video file",
            "detail": "Drag & drop or browse",
        },
        {
            "icon": "⚙️",
            "number": "2",
            "title": "Process",
            "description": "AI transcribes audio",
            "detail": "Fast & accurate",
        },
        {
            "icon": "🎯",
            "number": "3",
            "title": "Review",
            "description": "Synchronized playback",
            "detail": "Click to navigate",
        },
        {
            "icon": "💾",
            "number": "4",
            "title": "Export",
            "description": "Download results",
            "detail": "Multiple formats",
        },
    ]

    for col, step in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""
            <div class='step-card' style='
                position: relative;
                overflow: hidden;
            '>
                <div style='
                    font-size: 3rem;
                    margin-bottom: 0.5rem;
                '>{step['icon']}</div>
                <div style='
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: linear-gradient(
                        135deg,
                        var(--primary-gradient-start),
                        var(--primary-gradient-end)
                    );
                    color: white;
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: 0.9rem;
                '>
                    {step['number']}
                </div>
                <h3 style='
                    font-size: 1.3rem;
                    margin-bottom: 0.5rem;
                    color: var(--text-primary);
                '>
                    {step['title']}
                </h3>
                <p style='
                    color: var(--text-secondary);
                    margin-bottom: 0.25rem;
                    font-size: 0.95rem;
                '>
                    {step['description']}
                </p>
                <p class='step-subtext' style='
                    font-size: 0.85rem;
                    color: var(--text-muted);
                '>
                    {step['detail']}
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

    # Features section with enhanced cards
    st.markdown(
        """
    <h2 style='
        text-align: center;
        color: var(--text-primary);
        margin-bottom: 2rem;
        font-size: 1.8rem;
    '>
        ✨ Powerful Features
    </h2>
    """,
        unsafe_allow_html=True,
    )

    feature_col1, feature_col2 = st.columns(2)

    features_left = [
        {
            "icon": "⚡",
            "title": "Lightning Fast",
            "description": (
                "Powered by state-of-the-art Chunkformer ASR " "technology for rapid processing"
            ),
        },
        {
            "icon": "🎯",
            "title": "High Accuracy",
            "description": (
                "Optimized for Vietnamese language with " "industry-leading transcription quality"
            ),
        },
        {
            "icon": "🎬",
            "title": "Smart Sync",
            "description": (
                "Real-time transcript highlighting synchronized " "with video playback"
            ),
        },
    ]

    features_right = [
        {
            "icon": "📊",
            "title": "Multiple Formats",
            "description": (
                "Export to JSON (data), SRT (subtitles), " "or TXT (plain text) with one click"
            ),
        },
        {
            "icon": "🔄",
            "title": "Streaming Ready",
            "description": (
                "Handle videos of any length with intelligent " "chunk-based processing"
            ),
        },
        {
            "icon": "🎨",
            "title": "Modern UI",
            "description": (
                "Beautiful, intuitive interface with dark mode " "support and smooth animations"
            ),
        },
    ]

    with feature_col1:
        for feature in features_left:
            st.markdown(
                f"""
            <div style='
                background: var(--bg-tertiary);
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border-left: 4px solid var(--primary-gradient-start);
                transition: all 0.3s ease;
                border: 1px solid var(--border-color);
            '>
                <div style='
                    display: flex;
                    align-items: flex-start;
                    gap: 1rem;
                '>
                    <div style='
                        font-size: 2.5rem;
                        line-height: 1;
                    '>{feature['icon']}</div>
                    <div style='flex: 1;'>
                        <h4 style='
                            color: var(--text-primary);
                            margin-bottom: 0.5rem;
                            font-size: 1.1rem;
                        '>
                            {feature['title']}
                        </h4>
                        <p style='
                            color: var(--text-secondary);
                            font-size: 0.95rem;
                            margin: 0;
                            line-height: 1.6;
                        '>
                            {feature['description']}
                        </p>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with feature_col2:
        for feature in features_right:
            st.markdown(
                f"""
            <div style='
                background: var(--bg-tertiary);
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border-left: 4px solid var(--primary-gradient-start);
                transition: all 0.3s ease;
                border: 1px solid var(--border-color);
            '>
                <div style='
                    display: flex;
                    align-items: flex-start;
                    gap: 1rem;
                '>
                    <div style='
                        font-size: 2.5rem;
                        line-height: 1;
                    '>{feature['icon']}</div>
                    <div style='flex: 1;'>
                        <h4 style='
                            color: var(--text-primary);
                            margin-bottom: 0.5rem;
                            font-size: 1.1rem;
                        '>
                            {feature['title']}
                        </h4>
                        <p style='
                            color: var(--text-secondary);
                            font-size: 0.95rem;
                            margin: 0;
                            line-height: 1.6;
                        '>
                            {feature['description']}
                        </p>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

    # Supported formats section with badge design
    st.markdown(
        """
    <h2 style='
        text-align: center;
        color: var(--text-primary);
        margin-bottom: 2rem;
        font-size: 1.8rem;
    '>
        📋 Supported Video Formats
    </h2>
    """,
        unsafe_allow_html=True,
    )

    format_cols = st.columns(5)
    formats = [
        {"name": "MP4", "desc": "Most common"},
        {"name": "AVI", "desc": "High quality"},
        {"name": "MOV", "desc": "Apple format"},
        {"name": "MKV", "desc": "Open source"},
        {"name": "WebM", "desc": "Web optimized"},
    ]

    for col, fmt in zip(format_cols, formats):
        with col:
            st.markdown(  # noqa: E501
                f"""
            <div style='
                text-align: center;
                padding: 1.2rem 0.8rem;
                background: var(--bg-tertiary);
                border-radius: 12px;
                border: 2px solid var(--border-color);
                transition: all 0.3s ease;
                cursor: default;
            '
            onmouseover='
                this.style.transform="translateY(-4px)";
                this.style.borderColor="var(--primary-gradient-start)";
                this.style.boxShadow="0 8px 20px var(--shadow-md)";
            '
            onmouseout='
                this.style.transform="translateY(0)";
                this.style.borderColor="var(--border-color)";
                this.style.boxShadow="none";
            '>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>
                    🎥
                </div>
                <strong style='
                    color: var(--text-primary);
                    font-size: 1.1rem;
                    display: block;
                    margin-bottom: 0.25rem;
                '>
                    {fmt['name']}
                </strong>
                <span style='color: var(--text-muted); font-size: 0.8rem;'>
                    {fmt['desc']}
                </span>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

    # Tech specs / Additional info
    st.markdown(
        """
    <div style='
        background: linear-gradient(
            135deg,
            var(--bg-tertiary) 0%,
            var(--bg-secondary) 100%
        );
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid var(--border-color);
    '>
        <h3 style='
            color: var(--text-primary);
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
        '>
            💡 Why Choose Chunkformer?
        </h3>
        <div style='
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        '>
            <div>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>
                    🧠
                </div>
                <div style='
                    color: var(--text-primary);
                    font-weight: 600;
                    margin-bottom: 0.25rem;
                '>AI-Powered</div>
                <div style='
                    color: var(--text-muted);
                    font-size: 0.9rem;
                '>Advanced deep learning</div>
            </div>
            <div>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>
                    �
                </div>
                <div style='
                    color: var(--text-primary);
                    font-weight: 600;
                    margin-bottom: 0.25rem;
                '>Vietnamese Focus</div>
                <div style='
                    color: var(--text-muted);
                    font-size: 0.9rem;
                '>Language optimized</div>
            </div>
            <div>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>
                    🔒
                </div>
                <div style='
                    color: var(--text-primary);
                    font-weight: 600;
                    margin-bottom: 0.25rem;
                '>Privacy First</div>
                <div style='
                    color: var(--text-muted);
                    font-size: 0.9rem;
                '>Local processing</div>
            </div>
            <div>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>
                    ⚡
                </div>
                <div style='
                    color: var(--text-primary);
                    font-weight: 600;
                    margin-bottom: 0.25rem;
                '>Real-time</div>
                <div style='
                    color: var(--text-muted);
                    font-size: 0.9rem;
                '>Instant results</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_footer():
    """Render the footer section"""
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; padding: 2rem 0;'>
    <p class='hero-subtext' style='font-size: 0.9rem;'>
        Built with ❤️ using Chunkformer | <strong>Version 2.0</strong>
    </p>
    <p class='hero-subtext' style='font-size: 0.85rem;'>
        For support and documentation, visit the
        <a href='#' style='
            color: var(--primary-gradient-start);
            text-decoration: none;
        '>project repository</a>
    </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
