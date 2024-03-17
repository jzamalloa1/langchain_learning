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
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler # Specifically to stream agent tool output to container
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate

st.title("Agent Streamer :linked_paperclips:") # More icons at https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

def prompt_modifier():
    
    mod_prompt = hub.pull("hwchase17/openai-tools-agent")

    # system_prompt ="""
    # You are a helpful assistant. Assistant has access to the following tools: {tools}

    # To use a tool, please use the following format:

    # ```
    # Thought: Do I need to use a tool? Yes
    # Action: the action to take, should be one of [one of the tools provided]
    # Action Input: the input to the action
    # Observation: the result of the action
    # ```

    # When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    # ```
    # Thought: Do I need to use a tool? No
    # Final Answer: [your response here]
    # ```

    # """

    # mod_prompt.messages[0] = (
    #     SystemMessagePromptTemplate(prompt = PromptTemplate(input_variables=["tools"],
    #                                                         template=system_prompt))
    # )

    system_prompt = """
    You are a helpful assistant who will answer the User's questions. You will demonstrate your
    reasoning whenever you are answering a question. You have access to tools.
    You will try to ask questions to the user whenever you are comtemplating using a tool.
    You will ask for confirmation from the User whenever you are about to use a tool.
    """

    mod_prompt.messages[0] = (
        SystemMessagePromptTemplate(prompt = PromptTemplate(input_variables=[],
                                                            template=system_prompt))
    )

    return mod_prompt

def main():
    
    # Request Open API key on app start
    openai_api_key = st.sidebar.text_input('Enter OpenAI API Key', type='password')

    # Check if it is a valid Open API key and if so, continue with app
    if openai_api_key.startswith("sk-"):
        
        # Initiatize in session state default model
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-4-turbo-preview" # "gpt-3.5-turbo"

        # Initialize chat history and vector store in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I'm your AI assistant. How can I can help you?")
            ]
        
        # if "vector_store" not in st.session_state:
        #     st.session_state.vector_store

        # Initialize prompt, tools and streamlit callback (for agent tool streaming) in session state
        if "prompt" not in st.session_state:
            st.session_state.prompt = prompt_modifier()

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
                
                # Initalize st callback handler to stream tool output in container
                st_callback = StreamlitCallbackHandler(st.container())
                
                # Invoke agent as usual and add callback
                ai_response = agent_interaction_object.invoke(
                    {
                        "input":user_input,
                        "chat_history": st.session_state.chat_history, # Saved from session state
                    },
                    {
                        "callbacks": [st_callback]
                    }
                )
                
                st.write(ai_response["output"])
                # Get agent streaming chunks output on user's input
                # ai_response_stream = agent_interaction_object.stream(
                #     {
                #         "input":user_input,
                #         "chat_history": st.session_state.chat_history, # Saved from session state
                #     }
                # )

                # Render streaming response to AI's container
                # ai_response_content = st.write_stream(ai_response_stream) # NEEDS WORK
                # Update session state chat history with user's input exchange and agent's output (so it remains in history and we can render them)
                st.session_state.chat_history.extend(
                    [
                        HumanMessage(content=user_input),
                        AIMessage(content = ai_response["output"]) # NEEDS A LOT WORK
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

@tool
def px_trend_compare(data:str, col1:str, col2:str):
    """Only use this tool for plotly express stock data.
    Creates a line chart using plotly of two columns
    'col1' and 'col2,' which are two stock market ticker symbols,
    tracking their value over time. The data is
    available in 'data' stored in plotly express.
    Assign a different color for each line 'col1' and 'col2.'

    Args:
        data: Table name in plotly express
        col1: Stock market ticker symbol
        col2: Another stock market ticker symbol

    """

    import plotly.express as px

    data_module = getattr(px.data, data)
    df = data_module()

    df = (df
          .melt(id_vars="date", value_vars=[col1, col2])
    )

    fig = px.line(df, x='date', y='value', color='variable')
    
    return st.plotly_chart(fig)

def get_tool_box():
    
    tools = [where_cat_is_hiding, px_trend_compare]

    return tools

if __name__ == "__main__":

    # Call the main (orchestrating) function
    main()



