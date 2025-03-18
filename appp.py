import streamlit as st  # Must be first

# Set page config IMMEDIATELY after import
st.set_page_config(
    page_title="DS Tutor AI",
    page_icon="🧠",
    layout="centered"
)

# Now define other components
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain

API_KEY = "AIzaSyCHGvCV_UsrQLx8EZrb58IQ9qqQEyRNcYI"

# Custom CSS for chat messages
# In your existing CSS section, replace with:
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    .user-message { 
        background-color: #2d2d2d;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        color: white;
    }
    
    .bot-message { 
        background-color: #1a1a1a;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        color: white;
    }
    
    /* Adjust other UI elements */
    .st-bw, .stTextInput, .stNumberInput, .stSlider {
        background-color: #333333 !important;
        color: white !important;
    }
    
    .st-chat-input input {
        background-color: #333333 !important;
        color: white !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
# Corrected parameter

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# System prompt template
system_template = """You are an expert Data Science tutor..."""  # Keep your existing template

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Main chat interface
st.title("🧑🏫 AI Data Science Tutor")
st.subheader("Ask me anything about Data Science!")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="{msg["role"]}-message">{msg["content"]}</div>', 
                   unsafe_allow_html=True)  # Corrected here

# Chat input
if prompt_msg := st.chat_input("Ask your DS question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt_msg})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{prompt_msg}</div>', unsafe_allow_html=True)  # Corrected here

    # Initialize model and chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=API_KEY)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=st.session_state.memory,
        verbose=True
    )

    # Generate response
    with st.spinner("Analyzing your question..."):
        try:
            response = chain.invoke({"input": prompt_msg})["text"]
            
            # Add AI response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)  # Corrected here
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
