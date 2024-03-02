### GPT-App with Custom Agent Built-in (ON-HOLD while hashing out function calling vs custom tools)
# App will have streaming

import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools.retriever import create_retriever_tool
from langchain.output_parsers import JsonOutputToolsParser, JsonOutputKeyToolsParser
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub

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

        # Initialize prompt and tools in session state
        if "prompt" not in st.session_state:
            st.session_state.prompt = hub.pull("hwchase17/openai-tools-agent")

        if "tools" not in st.session_state:
            st.session_state.tools = get_tool_box()

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

        # Accept and render initial user input
        prompt = st.chat_input("Como quieres que El Gordo te ilumine hoy?")

        if prompt:

            # Render to user's container
            with st.chat_message("Human"):
                st.markdown(prompt)

            # Append to chat history in session state (to both have it in history and render)
            st.session_state.chat_history.append(
                HumanMessage(content=prompt)
                )
            
            # Get Agent to work....
            agent_response_object = get_ai_reponse(llm_model = llm, 
                                                   chat_history = st.session_state.chat_history,
                                                   prompt = st.session_state.prompt,
                                                   tool_box = st.session_state.tool_box)

    else:
        st.warning("Please enter an Open API Key to get started")

def get_ai_response(llm_model, chat_history, prompt, tool_box):

    # Create agent
    agent = create_openai_tools_agent(llm_model, tool_box, prompt)

    # Create agent executor
    agent_executor = AgentExecutor(agent=agent,
                                   tools=tool_box,
                                   verbose=True)

    # Interact with agent and get agent's response
    agent_output = agent_executor.invoke({
        "input":""
    })

    return agent_output

def get_tool_box():
    tools = []

    return tools

if __name__ == "__main__":

    # Call the main (orchestrating) function
    main()



