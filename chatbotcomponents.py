import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from datetime import timedelta
import re
from langchain_core.documents  import Document


class ChatbotComponents:
    """This class contains all the methods necesseory for Chatbot"""
    def __init__(self):
        pass
    
    
    def format_transcript_srt(self, transcription_list):
        """SRT-style - best for direct LLM Q&A"""
        lines = []
        for snippet in transcription_list.snippets:
            start_sec = int(snippet.start)
            minutes, second = divmod(start_sec, 60)
            timestamp = f"{minutes:02d}:{second:02d}"
            lines.append(f"[{timestamp}]: {snippet.text}")
        return " ".join(lines)
    
    
    def create_documents_to_store_in_vector_store(self, chunks):
        documents = []
        for chunk_text in chunks:
            match = re.search(r"\[(\d{1,3}:\d{2})\]", chunk_text)
            start_time = match.group(1) if match else None

            doc = Document(
                page_content = chunk_text,
                metadata={"start_time": start_time}
            )
            documents.append(doc)
        
        return documents
    
    
    def time_to_seconds(self, t):
        minutes, seconds = map(int, t.split(":"))
        return minutes * 60 + seconds
    

    def filter_by_time_range(self, documents, start, end):
        start_sec = self.time_to_seconds(start)
        end_sec = self.time_to_seconds(end)

        filtered_docs = []

        for doc in documents:
            if doc.metadata["start_time"]:
                doc_sec = self.time_to_seconds(doc.metadata["start_time"])

                if start_sec <= doc_sec <= end_sec:
                    filtered_docs.append(doc)

        return filtered_docs
    
    def rephrase_the_question(self, question, model):
        prompt = PromptTemplate(
            template="""
            You are an AI assistant.

            Rewrite the following user question to make it clearer, more specific, and well-structured 
            for retrieving relevant information from a video transcript.

            Rules:
            - Keep the original meaning exactly the same
            - Do NOT add new information
            - Make it concise and precise

            User Question:
            {question}
            """,
            input_variables=["question"]
        )

        chain = prompt | model | StrOutputParser()

        return chain.invoke({"question": question})
    


    def analyze_query_type(self, question, model):
        class QueryIntent(BaseModel):
            query_type: str = Field(description="One of: summarization, overview, qa")

        parser = PydanticOutputParser(pydantic_object=QueryIntent)

        template = PromptTemplate(
            template="""
                A user asked the following question while watching a YouTube video:

                {question}

                Classify the intent into exactly one of these three categories:

                - "summarization": The user explicitly wants a full summary, recap, TLDR, or brief of the entire video.
                  Trigger words: "summarize", "summary", "summarise", "recap", "tldr", "give me a brief".

                - "overview": The user wants to know what topics, subjects, or points are covered in the video — but is NOT asking for a full summary.
                  Trigger words: "list topics", "what topics", "what was discussed", "main topics", "subjects covered",
                  "what was talked about" (about the whole video), "what are the main points", "what subjects".

                - "qa": The user is asking a specific factual question about a person, event, or detail mentioned in the video.

                {format_instruction}
            """,
            input_variables=["question"],
            partial_variables={"format_instruction": parser.get_format_instructions()}
        )

        prompt = template.format(question=question)
        result = model.invoke(prompt)

        try:
            parsed = parser.parse(result.content)
            return parsed.query_type.strip().lower()
        except Exception:
            return "qa"
        
    
    def get_topic_overview(self, question, context_chunks, model):
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant analyzing a YouTube video transcript.

            Based on the transcript segments below, answer the user's question about what topics or subjects are covered.
            Format your answer as a clear bullet-point list with a one-line description for each topic.
            Only include topics that appear in the provided context. Do not hallucinate.

            --- CONTEXT START ---
            {context_chunks}
            --- CONTEXT END ---

            Question: {user_question}
            """,
            input_variables=["context_chunks", "user_question"]
        )
        chain = prompt | model | StrOutputParser()
        return chain.invoke({"context_chunks": context_chunks, "user_question": question})


    def summary_of_entire_video(self, chunks, model, batch_size=10):
        """
        Hierarchical summarization to avoid token limit errors.
        Summarizes chunks in batches, then combines batch summaries recursively.
        """
        # Step 1: Summarize individual chunks in batches
        batch_summaries = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            combined_batch = "\n".join(batch)
            summary = model.invoke(
                f"Summarize the following transcript excerpt concisely in 2-3 sentences:\n\n{combined_batch}"
            )
            batch_summaries.append(summary.content)

        # Step 2: Recursively combine until we have a single summary
        while len(batch_summaries) > 2*batch_size:
            next_level = []
            for i in range(0, len(batch_summaries), batch_size):
                group = batch_summaries[i:i + batch_size]
                combined = "\n".join(group)
                merged = model.invoke(
                    f"Combine these summaries into a single concise summary:\n\n{combined}"
                )
                next_level.append(merged.content)
            batch_summaries = next_level

        # Step 3: Final merge
        final_input = "\n".join(batch_summaries)
        final_summary = model.invoke(
            f"Write a clear, well-structured final summary of this YouTube video based on these notes:\n\n{final_input}"
        )
        return final_summary.content


    def compress_chat_history(self, chat_history, model):
        """Compress last 2 exchanges into a brief context string."""
        if not chat_history:
            return ""
        
        # Only use last 4 turns (2 exchange)
        recent = chat_history[-4:]
        history_text = "\n".join([f"{role}: {msg}" for role, msg in recent])

        prompt = PromptTemplate(
            template="""
            Compress the chat history into 2-3 sentences, keeping only facts that would help answer a follow-up question.
            Return Nothing if there's is NOTHING worth retaining.

            {history}
            """,
            input_variables=['history']
        )
        chain = prompt | model | StrOutputParser()

        return chain.invoke({"history": history_text})


    def get_answer(self, question, context_chunks, model, prior_context=""):
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant answering questions about a YouTube video.

            You will be given a set of context chunks extracted from the video transcript,
            each annotated with its timestamp in the format [MM:SS] or [HH:MM:SS].

            # Rules you must follow:
            1. Answer only from the provided context. Do not hallucinate.
            2. When you reference information, always cite the timestamp like:
            "At [02:34], the speaker explains that..."
            3. If the answer spans multiple segments, cite all relevant timestamps.
            4. If the context does not contain the answer, say:
            "This doesn't appear to be covered in the video."

            Relevant Prior Context (from earlier in this conversation):
            {prior_context}

            Relevant transcript segments:
            --- CONTEXT START ---
            {context_chunks}
            --- CONTEXT END ---

            My question: {user_question}
            """,
            input_variables=["context_chunks", "user_question", "prior_context"]
        )

        chain = prompt | model | StrOutputParser()

        return chain.invoke({"context_chunks": context_chunks, "user_question": question, "prior_context": prior_context})
    

    def initiate_chat(self, question, vector_store, heavy_model, light_model, chunks, documents, cached_summary=None, compressed_history=""):

        rephrase_question = self.rephrase_the_question(question, light_model)

        query_type = self.analyze_query_type(rephrase_question, light_model)

        if query_type == "summarization":
            if cached_summary:
                return cached_summary, None
            final_ans = self.summary_of_entire_video(chunks, light_model)
            return final_ans, final_ans

        elif query_type == "overview":
            overview_results = vector_store.similarity_search(query=rephrase_question, k=8)
            overview_context = "\n\n".join([doc.page_content for doc in overview_results])
            final_ans = self.get_topic_overview(rephrase_question, overview_context, light_model)
            return final_ans, None

        else:
            semantic_result = vector_store.similarity_search(query=rephrase_question, k=3)
            context_text = "\n\n".join([doc.page_content for doc in semantic_result])
            final_ans = self.get_answer(rephrase_question, context_text, light_model, prior_context=compressed_history)
            return final_ans, None