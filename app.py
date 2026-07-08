import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

CHROMA_PATH = "chroma"

# Page setup and design
st.set_page_config(page_title="Smart Fashion Assistant", page_icon="👗")
st.title("👗 Smart Fashion Assistant")
st.subheader("A Context-Aware RAG System for Personalized Fashion Styling & Body Shape Advising")

# 1. Formatting the browser's memory and conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Definition of Token
import os
import streamlit as st

HF_TOKEN = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face Token is missing! Please set HF_TOKEN environment variable.")
    st.stop()

# 2. Bringing in the database and the live client
@st.cache_resource
def init_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    client = InferenceClient(
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )
    return retriever, client

try:
    retriever, client = init_system()
except Exception as e:
    st.error("Failed to initialize the system, please check your internet connection and the API token.")
    st.stop()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Displaying previous messages in the chat interface
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)

# Receiving new questions from users
if user_query := st.chat_input("Let's design your look! Type your first fashion query here..."):
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    chat_history = st.session_state.chat_history
    
    with st.spinner("Consulting the fashion database... Your look is loading!"):
        # Retrieve the documents related to the question from Chroma Data
        relevant_docs = retriever.invoke(user_query)
        context = format_docs(relevant_docs)
        
        # Building Messages Compatible with Modern Dialogue Format
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI fashion stylist and a Proactive Conversational IR (CIR) assistant. "
                    "Your task is to answer the user's questions in English based ONLY on the following retrieved context:\n"
                    f"{context}\n\n"
                    "Strict Rules:\n"
                    "1. Rely strictly on the provided context. Do not hallucinate.\n"
                    "2. If the answer is not in the context, say: 'Based on the available fashion data, I do not have this information.'\n"
                    "3. [CRITICAL] To make the conversation highly interactive, ALWAYS end your response with ONE engaging follow-up or exploratory question. "
                    "This question should guide the user to explore more styling details from the documents (e.g., asking about their preferred colors, specific events, or matching accessories related to their body shape)."
                )
            }
        ]
        # Adding conversation history to maintain CIR context
        for msg in chat_history[-4:]:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            messages.append({"role": role, "content": msg.content})
            
        # Adding the current user question
        messages.append({"role": "user", "content": user_query})
        
       # Calling the modern interface with the supported model directly from Hugging Face servers
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            max_tokens=512,
            temperature=0.5
        )
        
        response_text = response.choices[0].message.content
        
    with st.chat_message("AI"):
        st.markdown(response_text)
        
    st.session_state.chat_history.extend([
        HumanMessage(content=user_query),
        AIMessage(content=response_text)
    ])