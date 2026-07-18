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

HF_TOKEN="add your tokrn from hugging face here"

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
        relevant_docs = retriever.invoke(user_query)
        context = format_docs(relevant_docs)
        
        # Building Messages Compatible with Modern Dialogue Format
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI fashion stylist and an advanced Proactive Conversational Information Retrieval (IR) assistant.\n"
                    "Your system operates STRICTLY in English. Your goal is to guide the user conversationally until you collect their profile across 4 specific pillars, and then recommend an outfit based ONLY on the provided fashion data.\n\n"
                    
                    f"Retrieved Fashion Data Context:\n{context}\n\n"
                    
                    "CONVERSATIONAL STRATEGY & PILLARS:\n"
                    "1. You must track exactly 4 pillars: Body Shape, Skin Tone, Occasion, and Weather/Season.\n"
                    "2. [STRICT MEMORY RULE]: Actively scan the user's current and previous inputs. If the user has ALREADY mentioned a pillar (e.g., if they said 'I have a job interview', then Occasion = Formal/Interview), you are strictly FORBIDDEN from asking about it again.\n"
                    "3. Identify only the REMAINING missing pillars. Acknowledge what the user said, and ask ONE natural follow-up question to discover only the information they haven't provided yet.\n"
                    "4. Do NOT provide the final outfit recommendation list until all 4 pillars are collected.\n"
                    "5. Once all 4 pillars are collected, provide a detailed, cohesive, and professional outfit recommendation based strictly on the retrieved context."
                                        
                    "CRITICAL TERMINOLOGY & TRANSLATION LAWS:\n"
                    "- Always use proper fashion terminology. NEVER output corrupted terms or literal machine translations (e.g., use 'Hourglass shape', NEVER 'glass watch'; use 'Wavy hair' or 'Professional updo', NEVER 'wifi').\n\n"
                    
                    "STRICT KNOWLEDGE-BASE MATCHING RULES:\n"
                    "When all 4 pillars are known, you must generate a highly polished, professional recommendation that strictly enforces the rules inside the retrieved data:\n"
                    "- [Match Occasion]: If the occasion is 'Formal / Interview' (OC004), you must strictly FORBID denim/jeans, sneakers, t-shirts, and casual wear. Recommend structured pantsuits or skirt suits in deep, conservative, professional colors (like navy blue, dark charcoal, or classic black). Strictly FORBID bright neon palettes, loud distracting patterns, or highly vibrant bottoms (like fuchsia or emerald green pants) for interviews, as the database mandates avoiding them for a serious impression.\n"
                    "- [Match Skin Tone with Occasion]: When matching colors for 'Dark Skin' (ST003) during a 'Formal / Interview', do NOT use fuchsia or vivid yellow for the main clothing. Instead, keep the high contrast by using a crisp white pressed button-down shirt/blouse (which perfectly complements dark skin and matches the interview code), paired with dark professional suits, and limit the elegant gold or bronze metallic tones strictly to simple accessories (like a watch or small earrings)."
                    "- [Match Body Shape]: For 'Hourglass' (BS001), focus on fitted tops and tailored pieces that highlight the waist; avoid oversized, boxy, or shapeless clothing. For other shapes (BS002 to BS006), adapt strictly according to their corresponding database entry text.\n"
                    "- [Match Weather/Season]: For Hot/Summer (SS002), recommend lightweight and breathable fabrics like cotton or linen, and avoid heavy solid blacks or dark charcoals. For Cold/Winter (SS001), recommend layered outfits, knitwear, and coats.\n"
                    
                    "OUTPUT FORMAT:\n"
                    "Output a beautifully structured, highly professional response in clear, fluent English that shows the final profile breakdown followed by the targeted clothing suggestions."
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
