import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import tempfile
from typing import Dict, List, Any

# Streamlit UI configuration
st.set_page_config(page_title="Legal Advice Chatbot", layout="wide")
st.title("📜 AskLawAI - Legal Advice Chatbot")
st.markdown("""
> 🛑 **Disclaimer:** This chatbot provides general legal information and explanations based on Indian law. It does **NOT** constitute legal advice or replace professional consultation.
""")

# Initialize session state for API keys if they don't exist
if 'api_keys_set' not in st.session_state:
    st.session_state.api_keys_set = False

# API key management
with st.sidebar:
    st.header("API Configuration")
    with st.expander("Configure API Keys", expanded=not st.session_state.api_keys_set):
        openai_api_key = st.text_input("OpenAI API Key:", type="password", help="Required for OpenAI models")
        gemini_api_key = st.text_input("Google AI (Gemini) API Key:", type="password", help="Required for Gemini models")
        
        if st.button("Save API Keys"):
            # Validate and save keys
            keys_valid = True
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            if gemini_api_key:
                os.environ["GOOGLE_API_KEY"] = gemini_api_key
            
            if openai_api_key or gemini_api_key:
                st.session_state.api_keys_set = True
                st.success("✅ API keys saved!")
            else:
                st.warning("⚠️ No API keys provided. Only Ollama models will be available.")
                st.session_state.api_keys_set = True

# Model type and selection
with st.sidebar:
    st.header("Model Configuration")
    model_provider = st.selectbox(
        "Select Model Provider:",
        options=["Ollama (Local)", "OpenAI", "Google Gemini"],
        index=0,
        help="Choose which AI provider to use"
    )
    
    # Dynamic model selection based on provider
    if model_provider == "Ollama (Local)":
        selected_model = st.selectbox(
            "Select Ollama Model:",
            options=["llama3.2", "llama3.1", "mistral", "gemma", "phi3"],
            index=0
        )
    elif model_provider == "OpenAI":
        if not openai_api_key and not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OpenAI API key required")
            selected_model = None
        else:
            selected_model = st.selectbox(
                "Select OpenAI Model:",
                options=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
                index=0
            )
    else:  # Google Gemini
        if not gemini_api_key and not os.getenv("GOOGLE_API_KEY"):
            st.error("⚠️ Google API key required")
            selected_model = None
        else:
            selected_model = st.selectbox(
                "Select Gemini Model:",
                options=["gemini-1.5-pro", "gemini-1.5-flash"],
                index=0
            )
    
    temperature = st.slider(
        "Temperature:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.2, 
        step=0.1,
        help="Higher values make output more creative, lower values more deterministic"
    )

# Helper function to get embedding model based on provider
def get_embeddings_model(provider: str, model_name: str) -> Any:
    if provider == "Ollama (Local)":
        return OllamaEmbeddings(model=model_name)
    elif provider == "OpenAI":
        return OpenAIEmbeddings(model="text-embedding-3-small")
    elif provider == "Google Gemini":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Helper function to get LLM model based on provider
def get_llm_model(provider: str, model_name: str, temp: float) -> Any:
    if provider == "Ollama (Local)":
        return Ollama(model=model_name, temperature=temp)
    elif provider == "OpenAI":
        return ChatOpenAI(model=model_name, temperature=temp)
    elif provider == "Google Gemini":
        return ChatGoogleGenerativeAI(model=model_name, temperature=temp)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Create tabs for document-specific and general advice
tab1, tab2 = st.tabs(["📄 Document Analysis", "🤔 General Legal Questions"])

with tab1:
    # Upload PDF Legal Document
    uploaded_file = st.file_uploader("Upload a contract or legal document (PDF)", type=["pdf"])
    
    if uploaded_file and selected_model:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        
        try:
            # Process document with progress indicators
            with st.spinner("Processing document..."):
                # Load and split document
                loader = PyPDFLoader(temp_path)
                pages = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(pages)
                
                # Create embeddings using selected provider
                embeddings = get_embeddings_model(model_provider, selected_model)
                vectorstore = FAISS.from_documents(docs, embeddings)
                
                # Set up RetrievalQA
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                
                st.success(f"✅ Document processed successfully! ({len(pages)} pages)")
                
                # Document information
                st.subheader("Document Overview")
                st.info(f"📑 Document contains {len(pages)} pages and has been split into {len(docs)} chunks for analysis.")
            
            # Custom prompt for legal analysis
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a helpful legal assistant with expertise in Indian contract law.
                Answer the user's question based on the context below.
                
                If the answer is not in the context, say "I'm not sure based on this document. Please consult a legal expert."
                
                Provide explanations in clear language that non-lawyers can understand.
                If relevant, mention specific clauses or sections from the document.
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            )
            
            # Initialize LLM based on selected provider
            llm = get_llm_model(model_provider, selected_model, temperature)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )
            
            # Input box
            query = st.text_input("Ask a legal question about the document:", 
                                placeholder="e.g., What does the indemnity clause mean? What are my obligations under this contract?")
            
            if query:
                with st.spinner("Analyzing and generating response..."):
                    result = qa_chain({"query": query})
                    
                    st.subheader("📘 Answer:")
                    st.markdown(result['result'])
                    
                    with st.expander("📄 Context from document"):
                        for i, doc in enumerate(result['source_documents']):
                            st.markdown(f"**Excerpt {i+1} - Page {doc.metadata['page']+1}:**")
                            st.markdown(f"```{doc.page_content[:800]}```")
        
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            if model_provider == "Ollama (Local)":
                st.info("⚠️ Make sure the Ollama service is running on your machine with the selected model.")
            elif model_provider == "OpenAI":
                st.info("⚠️ Check that your OpenAI API key is correct and has sufficient credits.")
            else:
                st.info("⚠️ Check that your Google API key is correct and properly configured.")
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    elif not selected_model:
        st.warning("⚠️ Please configure the required API key for the selected model provider.")
    
    else:
        st.info("📤 Upload a contract document above to get started.")

with tab2:
    st.subheader("🤔 General Legal Queries")
    st.markdown("Ask questions about Indian law without uploading a document.")
    
    with st.form("general_query_form"):
        general_query = st.text_area("Enter your legal question:", 
        placeholder="e.g., What IPC section applies to fraud? What are my rights as a tenant in Delhi?",
        height=100)
        submit_general = st.form_submit_button("Send")
    
    if general_query and selected_model:
        with st.spinner("Researching legal information..."):
            # Legal prompt template
            legal_prompt = f"""
            You are a legal assistant knowledgeable in Indian law, especially IPC (Indian Penal Code), civil law, and consumer protection.
            Answer the following question in clear, layman-friendly terms. 
            Include relevant IPC sections or legal statutes if applicable.
            Format your response with clear headings and bullet points where appropriate.
            
            Question: {general_query}
            
            Answer:
            """
            
            try:
                # Use selected model
                llm = get_llm_model(model_provider, selected_model, temperature)
                response = llm.invoke(legal_prompt)
                
                st.subheader("📘 Legal Information:")
                # Extract content based on model provider (different return types)
                if model_provider == "Ollama (Local)":
                    st.markdown(response)
                else:
                    st.markdown(response.content)
                st.warning("🛑 This is informational only and not a substitute for professional legal advice.")
            
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                if model_provider == "Ollama (Local)":
                    st.info("⚠️ Make sure the Ollama service is running on your machine with the selected model.")
                elif model_provider == "OpenAI":
                    st.info("⚠️ Check that your OpenAI API key is correct and has sufficient credits.")
                else:
                    st.info("⚠️ Check that your Google API key is correct and properly configured.")
    
    elif general_query and not selected_model:
        st.warning("⚠️ Please configure the required API key for the selected model provider.")

# Add a system status indicator
with st.sidebar:
    st.markdown("---")
    st.subheader("System Status")
    
    status_container = st.container()
    
    with status_container:
        # Check Ollama status (if selected)
        if model_provider == "Ollama (Local)":
            try:
                # Simple test to check if Ollama is running
                test_llm = Ollama(model=selected_model)
                test_response = test_llm.invoke("Hi")
                st.success("✅ Ollama service is running")
            except Exception:
                st.error("❌ Ollama service not detected")
                st.markdown("""
                **Troubleshooting:**
                1. Make sure Ollama is installed
                2. Ensure the Ollama service is running
                3. Check if the selected model is downloaded
                
                To install missing models run:
                ```
                ollama pull llama3.2
                ```
                """)
        
        # Check OpenAI API (if selected)
        elif model_provider == "OpenAI":
            if openai_api_key or os.getenv("OPENAI_API_KEY"):
                st.success("✅ OpenAI API key configured")
            else:
                st.error("❌ OpenAI API key not configured")
        
        # Check Gemini API (if selected)
        elif model_provider == "Google Gemini":
            if gemini_api_key or os.getenv("GOOGLE_API_KEY"):
                st.success("✅ Google API key configured")
            else:
                st.error("❌ Google API key not configured")

# Sidebar - System Info
with st.sidebar:
    if selected_model:
        st.markdown(f"**Model Provider:** {model_provider}")
        st.markdown(f"**Model:** {selected_model}")


# Footer
st.markdown("---")
st.markdown("Developed by Akshay Shekade 🤖 | Law-focused NLP Chatbot using LangChain with Ollama, OpenAI, and Google Gemini")

# Add helpful instructions
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### 🔍 Usage Tips
    
    **For Document Analysis:**
    - Upload PDF contracts or legal documents
    - Ask specific questions about clauses or terms
    - The system will find relevant sections
    
    **For General Questions:**
    - Ask about Indian legal concepts, IPC sections
    - Be specific about jurisdictions (state/city)
    - Specify the legal domain (criminal/civil/etc.)
    
    **Model Selection Tips:**
    - **Ollama models**: Run locally without API costs, but require local installation
    - **OpenAI models**: Highest quality responses, but require API credits
    - **Gemini models**: Good balance of quality and cost-efficiency
    """)