import streamlit as st
import openai
import chromadb
import PyPDF2
import requests
from datetime import datetime
import uuid
import json
import time
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import urllib.parse
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="🤖 AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for better UI
st.markdown(
    """
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .source-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "api_initialized" not in st.session_state:
    st.session_state.api_initialized = False
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False
if "search_results" not in st.session_state:
    st.session_state.search_results = []


class EnhancedRAGChatbot:
    def __init__(self):
        self.openai_client = None
        self.chroma_client = None
        self.collection = None
        self.serper_api_key = None

    def initialize_api(self, openai_key, serper_key=None):
        """Initialize OpenAI API and optionally Serper API"""
        try:
            # Initialize OpenAI
            self.openai_client = openai.OpenAI(api_key=openai_key)

            # Test the API key
            self.openai_client.models.list()

            # Initialize Serper API if provided
            if serper_key:
                self.serper_api_key = serper_key
                st.session_state.web_search_enabled = True

            # Initialize ChromaDB
            self.chroma_client = chromadb.Client()

            # Create or get collection
            try:
                self.collection = self.chroma_client.create_collection(
                    name="rag_documents", metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                self.collection = self.chroma_client.get_collection(
                    name="rag_documents"
                )

            return True
        except Exception as e:
            st.error(f"❌ Error initializing APIs: {str(e)}")
            return False

    def web_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Perform web search using Serper API"""
        if not self.serper_api_key:
            return []

        try:
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query, "num": num_results})
            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json",
            }

            response = requests.post(url, headers=headers, data=payload)
            if response.status_code == 200:
                data = response.json()
                results = []

                # Extract organic results
                if "organic" in data:
                    for result in data["organic"]:
                        results.append(
                            {
                                "title": result.get("title", ""),
                                "snippet": result.get("snippet", ""),
                                "link": result.get("link", ""),
                                "source": "web_search",
                            }
                        )

                return results
            else:
                st.error(f"Web search failed with status: {response.status_code}")
                return []

        except Exception as e:
            st.error(f"Web search error: {str(e)}")
            return []

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file with better error handling"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            total_pages = len(pdf_reader.pages)

            progress_bar = st.progress(0)
            for i, page in enumerate(pdf_reader.pages):
                text += page.extract_text() + "\n"
                progress_bar.progress((i + 1) / total_pages)

            progress_bar.empty()
            return text.strip()
        except Exception as e:
            st.error(f"❌ Error extracting text from PDF: {str(e)}")
            return None

    def extract_youtube_transcript(self, youtube_url):
        """Extract transcript from YouTube video with better URL parsing"""
        try:
            # Extract video ID from URL
            video_id = None
            if "youtube.com/watch?v=" in youtube_url:
                video_id = youtube_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in youtube_url:
                video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
            elif "youtube.com/embed/" in youtube_url:
                video_id = youtube_url.split("embed/")[1].split("?")[0]

            if not video_id:
                st.error("❌ Invalid YouTube URL format")
                return None

            # Get transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([entry["text"] for entry in transcript])

            # Get video title using basic approach
            try:
                video_title = f"YouTube Video ({video_id})"
                return text, video_title
            except:
                return text, f"YouTube Video ({video_id})"

        except Exception as e:
            st.error(f"❌ Error extracting YouTube transcript: {str(e)}")
            return None, None

    def extract_website_content(self, url):
        """Extract content from website with better parsing"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove unwanted elements
            for element in soup(
                ["script", "style", "nav", "footer", "header", "aside"]
            ):
                element.decompose()

            # Get title
            title = soup.title.string if soup.title else "Website Content"

            # Get main content
            main_content = (
                soup.find("main") or soup.find("article") or soup.find("body")
            )
            if main_content:
                text = main_content.get_text(separator=" ", strip=True)
            else:
                text = soup.get_text(separator=" ", strip=True)

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            return text, title

        except Exception as e:
            st.error(f"❌ Error extracting website content: {str(e)}")
            return None, None

    def generate_embedding(self, text):
        """Generate embedding using OpenAI's text-embedding-3-small model"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small", input=text, encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"❌ Error generating embedding: {str(e)}")
            return None

    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Split text into chunks for embedding"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    def add_to_vector_db(self, text, source_type, source_info):
        """Add text to ChromaDB vector database"""
        try:
            chunks = self.chunk_text(text)
            processed_chunks = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, chunk in enumerate(chunks):
                # Update progress
                progress = (i + 1) / len(chunks)
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {i + 1} of {len(chunks)}...")

                # Generate embedding using OpenAI
                embedding = self.generate_embedding(chunk)
                if embedding is None:
                    continue

                # Create unique ID
                doc_id = f"{source_type}_{source_info}_{i}_{uuid.uuid4().hex[:8]}"

                # Add to collection
                self.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[
                        {
                            "source_type": source_type,
                            "source_info": source_info,
                            "chunk_index": i,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ],
                    ids=[doc_id],
                )
                processed_chunks += 1

            progress_bar.empty()
            status_text.empty()

            # Add to processed docs
            st.session_state.processed_docs.append(
                {
                    "type": source_type,
                    "name": source_info,
                    "chunks": processed_chunks,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
            )

            st.success(
                f"✅ Successfully processed {processed_chunks} chunks from {source_type}: {source_info}"
            )
            return True
        except Exception as e:
            st.error(f"❌ Error adding to vector database: {str(e)}")
            return False

    def search_vector_db(self, query, n_results=5):
        """Search vector database for relevant chunks"""
        try:
            if not self.collection:
                return []

            # Generate query embedding using OpenAI
            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                return []

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )

            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            st.error(f"❌ Error searching vector database: {str(e)}")
            return []

    def generate_response(self, user_query, include_web_search=False):
        """Generate response using RAG with optional web search"""
        try:
            # Search vector database
            relevant_chunks = self.search_vector_db(user_query)

            # Perform web search if enabled and requested
            web_results = []
            if include_web_search and self.serper_api_key:
                web_results = self.web_search(user_query)
                st.session_state.search_results = web_results

            # Prepare context
            context = "\n".join(relevant_chunks) if relevant_chunks else ""

            # Add web search results to context
            if web_results:
                web_context = "\n".join(
                    [
                        f"Web Result: {result['title']} - {result['snippet']}"
                        for result in web_results
                    ]
                )
                context = f"{context}\n\nWeb Search Results:\n{web_context}"

            # Prepare prompt
            system_prompt = f"""You are a helpful AI assistant with access to various data sources and real-time web search. 
            Use the following context to answer the user's question comprehensively. If the context doesn't contain 
            relevant information, provide a general response and mention that more specific data might be needed.

            Context from knowledge base:
            {context}

            User Question: {user_query}

            Please provide a comprehensive and helpful response. If you used web search results, mention the sources."""

            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                max_tokens=1500,
                temperature=0.7,
            )
            return response.choices[0].message.content

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Please try again."


# Initialize chatbot
def get_chatbot():
    if st.session_state.chatbot is None:
        st.session_state.chatbot = EnhancedRAGChatbot()
    return st.session_state.chatbot


chatbot = get_chatbot()

# Sidebar - Enhanced API Configuration
with st.sidebar:
    st.header("🔧 Configuration")

    # API Keys Section
    with st.expander("🔑 API Keys", expanded=not st.session_state.api_initialized):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable AI responses",
            placeholder="sk-...",
        )

        serper_key = st.text_input(
            "Serper API Key (Optional)",
            type="password",
            help="Enter your Serper API key to enable web search",
            placeholder="Your Serper API key",
        )

        if openai_key:
            if st.button(
                "🚀 Initialize APIs", type="primary", use_container_width=True
            ):
                with st.spinner("Initializing APIs..."):
                    if chatbot.initialize_api(openai_key, serper_key):
                        st.session_state.vector_db = chatbot.chroma_client
                        st.session_state.collection = chatbot.collection
                        st.session_state.api_initialized = True
                        st.success("🎉 APIs initialized successfully!")
                        st.rerun()
        else:
            st.error("❌ OpenAI API Key Required")

    # Status Section
    st.header("📊 Status")

    # API Status
    if st.session_state.api_initialized:
        st.success("🟢 OpenAI: Connected")
        if st.session_state.web_search_enabled:
            st.success("🟢 Web Search: Enabled")
        else:
            st.info("🟡 Web Search: Disabled")
    else:
        st.error("🔴 APIs: Not Connected")

    # Database Stats
    if st.session_state.collection:
        try:
            doc_count = st.session_state.collection.count()
            st.metric("📚 Document Chunks", doc_count)
            st.metric("📁 Processed Sources", len(st.session_state.processed_docs))
        except:
            st.metric("📚 Document Chunks", 0)
            st.metric("📁 Processed Sources", 0)

    # Quick Actions
    st.header("⚡ Quick Actions")

    # Clear All Data with confirmation
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    if not st.session_state.confirm_clear:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning(
            "⚠️ Are you sure you want to clear all data? This action cannot be undone."
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes", use_container_width=True, type="primary"):
                if st.session_state.collection:
                    try:
                        # Get all document IDs and delete them
                        all_docs = st.session_state.collection.get()
                        if all_docs["ids"]:
                            st.session_state.collection.delete(ids=all_docs["ids"])

                        # Clear processed docs list
                        st.session_state.processed_docs = []
                        st.session_state.confirm_clear = False
                        st.success("All data cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing data: {str(e)}")
                        # Alternative approach - recreate collection
                        try:
                            # Delete and recreate collection
                            st.session_state.chroma_client.delete_scollection(
                                "rag_documents"
                            )
                            st.session_state.collection = (
                                st.session_state.chroma_client.create_collection(
                                    name="rag_documents",
                                    metadata={"hnsw:space": "cosine"},
                                )
                            )
                            st.session_state.processed_docs = []
                            st.session_state.confirm_clear = False
                            st.success("All data cleared (collection recreated)!")
                            st.rerun()
                        except Exception as e2:
                            st.error(f"Failed to clear data: {str(e2)}")
                            st.session_state.confirm_clear = False
                else:
                    st.session_state.confirm_clear = False

        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.search_results = []
        st.success("✅ Chat reset!")
        st.rerun()

# Main Content
st.title("🤖 AI Knowledge Assistant")
st.markdown("### Chat with your documents, websites, and the entire web!")

# Check if API is initialized
if not st.session_state.api_initialized:
    st.warning("⚠️ Please configure your OpenAI API key in the sidebar to get started.")
    st.info(
        "💡 **Pro Tip**: Add your Serper API key to enable web search capabilities!"
    )
    st.stop()

# Enhanced Tab Layout
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📚 Knowledge Base", "🔍 Web Search"])

with tab1:
    st.markdown("### 🎯 Intelligent Chat Interface")

    # Chat options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Ask questions about your uploaded documents or general topics")
    with col2:
        web_search_toggle = st.toggle(
            "🌐 Web Search",
            value=False,
            disabled=not st.session_state.web_search_enabled,
            help="Enable web search for current responses",
        )

    # Chat interface
    chat_container = st.container(height=450)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("💭 Ask me anything..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with chat_container.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                response = chatbot.generate_response(
                    prompt, include_web_search=web_search_toggle
                )
                st.markdown(response)

        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Chat controls
    # col1, col2, col3, col4 = st.columns(4)

    # with col1:
    #     pass

    # with col2:
    #     if st.session_state.chat_history:
    #         chat_export = "\n\n".join(
    #             [
    #                 f"**{msg['role'].title()}**: {msg['content']}"
    #                 for msg in st.session_state.chat_history
    #             ]
    #         )
    #         st.download_button(
    #             label="📥 Export Chat",
    #             data=chat_export,
    #             file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    #             mime="text/plain",
    #             use_container_width=True,
    #         )

    # with col3:
    #     pass

    # with col4:
    #     if st.session_state.search_results:
    #         st.metric("🔍 Web Results", len(st.session_state.search_results))

with tab2:
    st.markdown("### 📚 Knowledge Base Management")

    # Data source upload interface
    st.markdown("#### 📤 Add New Data Sources")

    # Enhanced upload interface
    source_tabs = st.tabs(
        ["📄 PDF Documents", "🎥 YouTube Videos", "🌐 Websites", "📝 Text Input"]
    )

    with source_tabs[0]:
        st.markdown("##### Upload PDF Documents")
        pdf_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload multiple PDF documents to extract and analyze their content",
        )

        if pdf_files:
            st.success(f"📎 {len(pdf_files)} PDF(s) selected")

            if st.button(
                "🔄 Process All PDFs", key="process_pdfs", use_container_width=True
            ):
                for pdf_file in pdf_files:
                    with st.spinner(f"📖 Processing {pdf_file.name}..."):
                        text = chatbot.extract_text_from_pdf(pdf_file)
                        if text:
                            chatbot.add_to_vector_db(text, "PDF", pdf_file.name)
                st.rerun()

    with source_tabs[1]:
        st.markdown("##### Extract YouTube Transcripts")
        youtube_urls = st.text_area(
            "YouTube URLs (one per line)",
            placeholder="https://youtube.com/watch?v=...\nhttps://youtu.be/...",
            help="Enter YouTube URLs to extract transcripts",
        )

        if youtube_urls:
            urls = [url.strip() for url in youtube_urls.split("\n") if url.strip()]
            st.info(f"🎬 {len(urls)} URL(s) entered")

            if st.button(
                "🔄 Process Videos", key="process_videos", use_container_width=True
            ):
                for url in urls:
                    with st.spinner(f"🎬 Processing {url}..."):
                        result = chatbot.extract_youtube_transcript(url)
                        if result and len(result) == 2:
                            text, title = result
                            chatbot.add_to_vector_db(text, "YouTube", title)
                st.rerun()

    with source_tabs[2]:
        st.markdown("##### Scrape Website Content")
        website_urls = st.text_area(
            "Website URLs (one per line)",
            placeholder="https://example.com\nhttps://blog.example.com/post",
            help="Enter website URLs to extract content",
        )

        if website_urls:
            urls = [url.strip() for url in website_urls.split("\n") if url.strip()]
            st.info(f"🌐 {len(urls)} URL(s) entered")

            if st.button(
                "🔄 Process Websites", key="process_websites", use_container_width=True
            ):
                for url in urls:
                    with st.spinner(f"🌍 Processing {url}..."):
                        result = chatbot.extract_website_content(url)
                        if result and len(result) == 2:
                            text, title = result
                            chatbot.add_to_vector_db(text, "Website", title)
                st.rerun()

    with source_tabs[3]:
        st.markdown("##### Direct Text Input")
        text_input = st.text_area(
            "Enter text directly",
            placeholder="Paste your text content here...",
            height=200,
            help="Enter text directly to add to knowledge base",
        )

        text_title = st.text_input(
            "Title for this text", placeholder="Give your text a descriptive title"
        )

        if text_input and text_title:
            if st.button("🔄 Add Text", key="process_text", use_container_width=True):
                with st.spinner("📝 Processing text..."):
                    chatbot.add_to_vector_db(text_input, "Text", text_title)
                st.rerun()

    # Display processed documents
    st.markdown("---")
    st.markdown("#### 📁 Processed Documents")

    if st.session_state.processed_docs:
        for i, doc in enumerate(st.session_state.processed_docs):
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.markdown(f"**{doc['name']}**")
                    st.caption(f"Type: {doc['type']}")
                with col2:
                    st.metric("Chunks", doc["chunks"])
                with col3:
                    st.caption(f"Added: {doc['timestamp']}")
                with col4:
                    if st.button(
                        "🗑️",
                        key=f"delete_{i}_{doc['name']}",
                        help="Delete this document",
                    ):
                        try:
                            # Find and delete chunks related to this document
                            if st.session_state.collection:
                                # Get all documents with matching source_info
                                all_docs = st.session_state.collection.get()

                                # Find IDs of chunks belonging to this document
                                ids_to_delete = []
                                for j, metadata in enumerate(all_docs["metadatas"]):
                                    if (
                                        metadata.get("source_info") == doc["name"]
                                        and metadata.get("source_type") == doc["type"]
                                    ):
                                        ids_to_delete.append(all_docs["ids"][j])

                                # Delete the chunks
                                if ids_to_delete:
                                    st.session_state.collection.delete(
                                        ids=ids_to_delete
                                    )
                                    st.success(
                                        f"✅ Deleted {len(ids_to_delete)} chunks"
                                    )

                            # Remove from processed docs
                            st.session_state.processed_docs.remove(doc)
                            st.success(f"✅ Removed {doc['name']} from knowledge base")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting document: {str(e)}")
                            # Still remove from processed docs list
                            st.session_state.processed_docs.remove(doc)
                            st.rerun()
    else:
        st.info("📭 No documents processed yet. Upload some content to get started!")

with tab3:
    st.markdown("### 🔍 Web Search Results")

    if not st.session_state.web_search_enabled:
        st.warning(
            "⚠️ Web search is disabled. Please add your Serper API key in the sidebar to enable this feature."
        )
    else:
        st.success("🌐 Web search is enabled!")

        # Manual search interface
        st.markdown("#### 🔎 Manual Web Search")

        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Search the web",
                placeholder="Enter your search query...",
                help="Search the web for real-time information",
            )
        with col2:
            num_results = st.selectbox("Results", [3, 5, 10], index=1)

        if search_query:
            if st.button("🔍 Search Web", use_container_width=True):
                with st.spinner("🌐 Searching the web..."):
                    results = chatbot.web_search(search_query, num_results)
                    st.session_state.search_results = results
                st.rerun()

    # Display search results
    if st.session_state.search_results:
        st.markdown("#### 📋 Latest Search Results")

        for i, result in enumerate(st.session_state.search_results):
            with st.container():
                st.markdown(f"**{i + 1}. {result['title']}**")
                st.markdown(result["snippet"])
                st.markdown(f"🔗 [Read More]({result['link']})")
                st.markdown("---")
    else:
        st.info(
            "🔍 No search results yet. Use the chat with web search enabled or perform a manual search above."
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <p>🚀 <strong>AI Knowledge Assistant</strong> | Powered by OpenAI GPT-3.5 • ChromaDB • Serper API</p>
    </div>
    """,
    unsafe_allow_html=True,
)
