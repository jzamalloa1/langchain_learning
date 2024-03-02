### GPT-App with Custom Agent Built-in (ON-HOLD while hashing out function calling vs custom tools)
# App will have streaming

import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent, AgentExecutor


st.title("Agent Streamer :linked_paperclips:") # More icons at https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

def main():
    
    # Request Open API key on app start
    openai_api_key = st.sidebar.text_input('Enter OpenAI API Key', type='password')

    # Check if it is a valid Open API key and if so, continue with app
    if openai_api_key.startswith("sk-"):
        
        # Initialize chat history and vector store in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I'm your AI assistant. How can I can help you?")
            ]
        
        # if "vector_store" not in st.session_state:
        #     st.session_state.vector_store
            
        # Initialize LLM model
        llm = ChatOpenAI(model=st.session_state["openai_model"], 
                        temperature=0.1, streaming=True, api_key=openai_api_key)
        
        # Render chat messages in role containers
        for m in st.session_state["chat_history"]:
            if isinstance(m, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(m["content"])
            elif isinstance(m, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(m["content"])

    else:
        st.warning("Please enter an Open API Key to get started")


if __name__ == "__main__":

    # Call the main (orchestrating) function
    main()



