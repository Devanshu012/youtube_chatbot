# 🎥 YouTube Video Chatbot

A RAG-powered chatbot that lets you have intelligent conversations about any YouTube video using its transcript. Ask questions, get summaries, or explore topics — all grounded in what the video actually says.

---

## ✨ Features

- **Q&A** — Ask specific questions about the video and get timestamped answers
- **Full Summarization** — Get a concise, hierarchical summary of the entire video
- **Topic Overview** — List all the key topics and subjects covered in the video
- **Context-Aware Chat** — Compressed chat history keeps conversations coherent over multiple turns
- **Smart Query Routing** — Automatically detects whether your question needs a summary, overview, or direct answer
- **Caching** — Video summaries are cached to avoid redundant LLM calls

---

## 🧠 How It Works

```
YouTube URL → Transcript Fetch → Chunking → Vector Store (Chroma)
                                                    ↓
User Question → Query Rephrasing → Intent Classification
                                          ↓
                        ┌─────────────────┼──────────────────┐
                   Summarization       Overview              Q&A
                        ↓                 ↓                   ↓
               Hierarchical         Similarity           Semantic Search
               Summarization         Search           + Timestamped Answer
```

1. The transcript is fetched and split into overlapping chunks
2. Chunks are embedded and stored in a **Chroma** vector store
3. Each user question is rephrased for clarity, then classified by intent
4. The appropriate pipeline (summarization / overview / Q&A) is triggered
5. Chat history is periodically compressed to maintain context without bloating the prompt

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| LLM | [Groq](https://console.groq.com/) (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) |
| Embeddings | [HuggingFace](https://huggingface.co/) (`all-MiniLM-L6-v2`) |
| Vector Store | [Chroma](https://www.trychroma.com/) |
| Transcript | [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) |
| Orchestration | [LangChain](https://www.langchain.com/) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/youtube-video-chatbot.git
cd youtube-video-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Use the app

1. Enter your **Groq API Key** in the sidebar (get one free at [console.groq.com](https://console.groq.com/))
2. Paste a **YouTube URL or Video ID**
3. Click **📥 Load Video**
4. Start chatting!

---

## 📦 Requirements

```
streamlit
langchain
langchain-groq
langchain-huggingface
langchain-chroma
langchain-text-splitters
langchain-core
sentence-transformers
youtube-transcript-api
chromadb
pydantic
```

> **Note:** A free Groq API key is all you need — no OpenAI billing required.

---

## 📁 Project Structure

```
youtube-video-chatbot/
│
├── app.py                  # Main Streamlit app
├── chatbotcomponents.py    # Core chatbot logic (RAG pipelines, LLM chains)
├── requirements.txt
└── README.md
```

---

## 💡 Example Questions

Once a video is loaded, try asking:

- *"Summarize this video"*
- *"What topics are covered?"*
- *"What did the speaker say about X at the beginning?"*
- *"Explain the part about Y in simple terms"*

---

## ⚠️ Limitations

- Only works with videos that have publicly available transcripts (auto-generated or manual captions)
- Very long videos may take a moment to summarize due to hierarchical chunking
- Transcript quality depends on YouTube's captioning accuracy

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [Groq](https://groq.com/) for blazing-fast LLM inference
- [LangChain](https://langchain.com/) for the RAG framework
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) for easy transcript access