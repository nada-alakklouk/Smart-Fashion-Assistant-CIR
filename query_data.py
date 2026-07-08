import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# Page setup and design
st.set_page_config(page_title="Smart Fashion Assistant (CIR)", page_icon="👗")
st.title("👗 Smart Fashion Assistant (CIR)")
st.subheader("Academic Conversational Information Retrieval System (CIR)")

# 1. Formatting the browser's memory and conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 2. Retrieving the database and LLM
@st.cache_resource
def init_system():
    embeddings = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2}) # Retrieve the best matching chanks
    
    #We will use GPT for instant connection,
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    return retriever, llm

try:
    retriever, llm = init_system()
except Exception as e:
    st.error(" Make sure you have set the OPENAI_API_KEY in the .env file or system settings to generate responses.")
    st.stop()

# 3. Building the Conversational Pipeline and Context Tracking (According to SIGIR 2020 Slides)
# Merging the new question with the previous conversation for rephrasing
contextualize_q_system_prompt = (
    "Considering the conversation history and the user's latest question, "
    "rephrase the question into a standalone query that can be understood without the conversation history. "
    "Do not answer the question, just rephrase it if necessary, otherwise leave it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# Integrate the retrieved context to generate the final answer
qa_system_prompt = (
    "You are a knowledgeable fashion expert and conversational assistant (CIR). Answer the user's questions "
    "based only on the following retrieved information and extracted from our fashion files:\n\n"
    "{context}\n\n"
    "If the answer is not present in the context, clearly state that you do not have this information available."
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Display previous messages in the chat interface
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)

# Receive the new question from the user
if user_query := st.chat_input("Ask me about fashion styling tips and body types..."):
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    # Extract the current conversation history
    chat_history = st.session_state.chat_history
    
    # Dialogue tracking: Formulating a semantically contextual question
    with st.spinner("Context tracking and data retrieval are underway..."):
        if chat_history:
            contextualized_query = contextualize_q_chain.invoke({"input": user_query, "chat_history": chat_history})
        else:
            contextualized_query = user_query
            
       # Searching in 17 Shanks based on the smart question
        relevant_docs = retriever.invoke(contextualized_query)
        context = format_docs(relevant_docs)
        
        # Generating the final dialogic response
        rag_chain = qa_prompt | llm
        response = rag_chain.invoke({"input": user_query, "context": context, "chat_history": chat_history})
    
    with st.chat_message("AI"):
        st.markdown(response.content)
        
    # Save the current dialogue in memory for the next round
    st.session_state.chat_history.extend([
        HumanMessage(content=user_query),
        AIMessage(content=response.content)
    ])