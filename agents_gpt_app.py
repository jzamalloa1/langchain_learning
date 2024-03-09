### GPT-App with Custom Agent Built-in (ON-HOLD while hashing out function calling vs custom tools)
# App will have streaming

import streamlit as st
import pandas as pd
import numpy as np
import random
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools.retriever import create_retriever_tool
from langchain.output_parsers import JsonOutputToolsParser, JsonOutputKeyToolsParser
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain import hub
from langchain.tools import tool

st.title("Agent Streamer :linked_paperclips:") # More icons at https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

def main():
    
    # Request Open API key on app start
    openai_api_key = st.sidebar.text_input('Enter OpenAI API Key', type='password')

    # Check if it is a valid Open API key and if so, continue with app
    if openai_api_key.startswith("sk-"):
        
        # Initiatize in session state default model
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

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

        if "tool_box" not in st.session_state:
            st.session_state.tool_box = get_tool_box()

        # Initialize LLM
        llm = ChatOpenAI(model=st.session_state["openai_model"], 
                        temperature=0.1, streaming=True, api_key=openai_api_key)
        
        # Bind tools to LLM 
        llm_bound_tools = llm.bind_tools(st.session_state.tool_box)

        # Render chat messages in role containers
        for m in st.session_state.chat_history:
            if isinstance(m, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(m.content)
            elif isinstance(m, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(m.content)

        # Accept and render initial user input
        user_input = st.chat_input("Como quieres que El Gordo te ilumine hoy?")

        if user_input:

            # Render user's query exchange to user's container in Chat
            with st.chat_message("Human"):
                st.markdown(user_input)
            
            # Get Agent to work. Include chat history
            agent_interaction_object = get_ai_executor(llm_with_tools = llm_bound_tools, 
                                                       prompt = st.session_state.prompt,
                                                       tool_box = st.session_state.tool_box)

            # Render AI response to AI's container Chat
            with st.chat_message("AI"):

                # Get agent streaming chunks output on user's input
                ai_response_stream = agent_interaction_object.stream(
                    {
                        "input":user_input,
                        "chat_history": st.session_state.chat_history, # Saved from session state
                    }
                )

                # Render streaming response to AI's container
                ai_response_content = st.write_stream(ai_response_stream) # NEEDS WORK

                # Update session state chat history with user's input exchange and agent's output (so it remains in history and we can render them)
                st.session_state.chat_history.extend(
                    [
                        HumanMessage(content=user_input),
                        AIMessage(content = ai_response_content[-1]["output"]) # NEEDS A LOT WORK
                    ]
                )

    else:
        st.warning("Please enter an Open API Key to get started")

def get_ai_executor(llm_with_tools, prompt, tool_box):

    # Create agent
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]), # Part of the prompt as well
            "chat_history": lambda x: x["chat_history"] # Part of the prompt as well
        }
        | prompt
        | llm_with_tools.with_config({"tags": ["agent_llm"]}) # FOR STREAMING
        | OpenAIToolsAgentOutputParser()
    )

    # Create agent executor
    agent_executor = AgentExecutor(agent=agent,
                                   tools=tool_box,
                                   verbose=True).with_config({"run_name":"Agent"}) # FOR STREAMING

    return agent_executor

@tool
def where_cat_is_hiding() -> str:
    """Where is the cat hiding right now?"""
    return random.choice(["under the bed", "on the shelf"])


def get_tool_box():
    
    tools = [where_cat_is_hiding]

    return tools

if __name__ == "__main__":

    # Call the main (orchestrating) function
    main()



