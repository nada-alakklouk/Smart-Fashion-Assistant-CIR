import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from custom_embeddings import CustomEmbeddings
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

HF_TOKEN="add your token here"

if not HF_TOKEN:
    st.error("Hugging Face Token is missing! Please set HF_TOKEN environment variable.")
    st.stop()

# 2. Bringing in the database and the live client
@st.cache_resource
def init_system():
    embeddings = CustomEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    # اجعلي السطر بهذا الشكل تماماً:
    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)
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
        search_query = user_query
        if chat_history:
            search_query = f"{chat_history[0].content} {user_query}"
            
        relevant_docs = retriever.invoke(search_query)
        context = format_docs(relevant_docs)
        
        # Building Messages Compatible with Modern Dialogue Format
        system_content = (
            "You are an expert AI fashion stylist and an advanced Proactive Conversational Information Retrieval (IR) assistant.\n"
            "Your system operates STRICTLY in English. Your goal is to guide the user conversationally until you collect their profile across 4 specific pillars, and then recommend an outfit based ONLY on the provided fashion data.\n\n"
            
            f"Retrieved Fashion Data Context:\n{context}\n\n"
            
            "CONVERSATIONAL STRATEGY & SLOT TRACKING:\n"
            "1. You must carefully track exactly 4 pillars: [Occasion], [Body Shape], [Weather/Season], and [Skin Tone].\n"
            "2. [STRICT AMBIGUITY GUARD]: If the user provides a vague, non-standard, or generic answer for any pillar "
            "(e.g., body shape is 'normal', 'regular', 'average', or 'I don't know'), you are strictly FORBIDDEN from accepting it "
            "as a valid slot value. You must explicitly reject it, mark it as [Missing/Ambiguous], and rephrase your question.\n"
            "3. [DYNAMIC REPHRASING & CLARIFICATION]: When a pillar is [Missing/Ambiguous], acknowledge what the user said, "
            "explain why that answer cannot determine a garment fit, and rephrase your follow-up by presenting specific, clear options "
            "from your system (e.g., for body shape: Hourglass, Inverted Triangle, Pear, Apple).\n"
            "4. [DIALOG MANAGEMENT]: Scan current and previous turns. Do NOT ask for information already clearly provided. "
            "Ask only ONE natural, high-quality clarifying question at a time to resolve the remaining [Missing/Ambiguous] pillars.\n"
            "5. [STRICT BLOCKER]: Do NOT output the final outfit recommendation or retrieve formatting layouts until ALL 4 pillars "
            "are specifically and validly collected.\n\n"
            
            "OUTPUT FORMATTING:\n"
            "For your response, always maintain a polite, conversational tone. If all pillars are met, output the final cohesive recommendation.\n\n"
            
            "CRITICAL TERMINOLOGY & TRANSLATION LAWS:\n"
            "- Always use proper fashion terminology. NEVER output corrupted terms or literal machine translations (e.g., use 'Hourglass shape', NEVER 'glass watch').\n\n"
            
            "STRICT KNOWLEDGE-BASE MATCHING RULES:\n"
            "When all 4 pillars are known, generate a highly polished recommendation enforcing retrieved rules:\n"
            "- [Match Occasion]: If 'Formal / Interview' (OC004), strictly FORBID denim/jeans, sneakers, t-shirts. Recommend structured pantsuits/skirtsuits in navy blue, charcoal, or black. Strictly FORBID bright neon or loud patterns.\n"
            "- [Match Skin Tone with Occasion]: For 'Dark Skin' (ST003) in an interview, do NOT use fuchsia or vivid yellow for main clothing. Use a crisp white blouse/shirt for high contrast, and limit gold/bronze metallic tones strictly to simple accessories.\n"
            "- [Match Body Shape]: For 'Hourglass' (BS001), focus on fitted tops and tailored pieces highlighting the waist; avoid boxy/shapeless clothing.\n"
            "- [Match Weather/Season]: For Hot/Summer (SS002), recommend lightweight and breathable fabrics like cotton or linen.\n\n"
            
            "OUTPUT FORMAT:\n"
            "Output a beautifully structured, highly professional response in clear, fluent English showing the profile breakdown followed by targeted clothing suggestions."
        )
        
        messages = [{"role": "system", "content": system_content}]
        # Adding conversation history to maintain CIR context
        for msg in chat_history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            messages.append({"role": role, "content": msg.content})
            
        # Adding the current user question
        messages.append({"role": "user", "content": user_query})
        
       # Calling the modern interface with the supported model directly from Hugging Face servers
        response = client.chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.5
        )

        response_text = response.choices[0].message.content
        
    with st.chat_message("AI"):
        st.markdown(response_text)
        
    st.session_state.chat_history.extend([
        HumanMessage(content=user_query),
        AIMessage(content=response_text)
    ])
