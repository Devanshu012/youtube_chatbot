import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from youtube_transcript_api import YouTubeTranscriptApi
import re

from chatbotcomponents import ChatbotComponents

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="🎥 YouTube Chatbot",
    page_icon="🤖",
    layout="wide"
)

def extract_video_id(url_or_id: str):
    """
    Extract YouTube video ID from URL
    """
    patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",        # normal URL
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})", # short URL
        r"(?:embed/)([0-9A-Za-z_-]{11})"      # embed URL
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    # fallback → assume user pasted ID directly
    if len(url_or_id) == 11:
        return url_or_id

    return None

# ------------------- SIDEBAR -------------------
st.sidebar.title("⚙️ Settings")

# API Key input
groq_api_key = st.sidebar.text_input(
    "🔑 Groq API Key",
    type="password",
    placeholder="gsk_...",
    help="Get your free key at console.groq.com"
)

if groq_api_key:
    st.sidebar.success("✅ API key entered")
else:
    st.sidebar.warning("⚠️ Enter your Groq API key to start")

video_input = st.sidebar.text_input(
    "Enter YouTube URL or Video ID",
    value="",
    disabled=not groq_api_key
)

load_button = st.sidebar.button("📥 Load Video", disabled=not groq_api_key)

# ------------------- MODELS -------------------
@st.cache_resource
def load_models(api_key: str):
    light_model = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key
    )

    heavy_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return light_model, heavy_model, embedding_model

if groq_api_key:
    light_model, heavy_model, embedding_model = load_models(groq_api_key)
else:
    light_model, heavy_model, embedding_model = None, None, None

bot = ChatbotComponents()

# ------------------- SESSION STATE -------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "documents" not in st.session_state:
    st.session_state.documents = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "cached_summary" not in st.session_state:
    st.session_state.cached_summary = None

if "compressed_history" not in st.session_state:
    st.session_state.compressed_history = ""

# ------------------- LOAD VIDEO -------------------
if load_button:
    video_id = extract_video_id(video_input)

    if not video_id:
        st.error("❌ Invalid YouTube URL or Video ID")
    else:
        with st.spinner("Fetching transcript..."):
            try:
                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.fetch(video_id)

                transcript = bot.format_transcript_srt(transcript_list)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=50
                )
                chunks = splitter.split_text(transcript)

                documents = bot.create_documents_to_store_in_vector_store(chunks)

                vector_store = Chroma(
                    embedding_function=embedding_model,
                    persist_directory="my_chroma_db",
                    collection_name='sample'
                )

                vector_store.reset_collection()
                vector_store.add_documents(documents)

                st.session_state.vector_store = vector_store
                st.session_state.chunks = chunks
                st.session_state.documents = documents
                st.session_state.chat_history = []
                st.session_state.cached_summary = None
                st.session_state.compressed_history = ""

                st.success("✅ Video loaded successfully!")

                # 🔥 Show embedded video
                st.video(f"https://www.youtube.com/watch?v={video_id}")

            except Exception as e:
                st.error(f"Error: {e}")

# ------------------- MAIN UI -------------------
st.title("🎥 YouTube Video Chatbot")
st.markdown("Ask questions about the video or request summaries.")

# ------------------- CHAT DISPLAY -------------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# ------------------- INPUT -------------------
user_input = st.chat_input("Ask something about the video...", disabled=not groq_api_key)

if user_input:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please load a video first!")
    else:
        # Show user message
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                try:
                    response, new_summary = bot.initiate_chat(
                        user_input,
                        st.session_state.vector_store,
                        heavy_model,
                        light_model,
                        st.session_state.chunks,
                        st.session_state.documents,
                        st.session_state.cached_summary,
                        compressed_history=st.session_state.compressed_history
                    )

                    st.markdown(response)
                    st.session_state.chat_history.append(("assistant", response))
                    if len(st.session_state.chat_history) % 4 == 0:
                        st.session_state.compressed_history = bot.compress_chat_history(
                            st.session_state.chat_history, light_model
                        )

                    if new_summary:
                        st.session_state.cached_summary = new_summary

                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(("assistant", error_msg))