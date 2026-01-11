import streamlit as st
from openai import OpenAI
import time

# Page config
st.set_page_config(
    page_title="Local LLM Chat",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for configuration
with st.sidebar:
    st.title("Settings ⚙️")
    st.markdown("This app uses a local LLM running via [Ollama](https://ollama.com).")
    
    # Model selection (you can add more models here)
    model_option = st.selectbox(
        "Select Model",
        ("llama3.2", "llama3", "mistral", "gemma"),
        index=0
    )
    
    st.divider()
    
    # Connection status check
    try:
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )
        # Simple check to see if we can list models (implies connection is up)
        # Note: This might be slow if the server is sleeping, but good for a status check
        models = client.models.list()
        st.success(f"Connected to Ollama! 🟢")
        # Optional: update model list dynamically
        # available_models = [m.id for m in models.data]
    except Exception as e:
        st.error(f"Could not connect to Ollama. Make sure it's running via `ollama serve`. 🔴")
        st.info("Download Ollama from [ollama.com](https://ollama.com)")

# Main chat interface
st.title("💬 Local LLM Chat")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
            
            stream = client.chat.completions.create(
                model=model_option,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error communicating with model: {e}")
            st.info("ensure you have pulled the model using `ollama pull <model_name>`")

