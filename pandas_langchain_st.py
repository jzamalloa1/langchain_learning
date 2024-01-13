from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import streamlit as st
import pandas as pd
import numpy as np
import os

############# To review Streamlit's session state: https://docs.streamlit.io/library/api-reference/session-state #############
# st.set_page_config(page_title="Langchain agent 'El Bryan' - A pandas AI", page_icon="🦜")
# st.title("🦜 Langchain agent 'El Bryan' - A pandas AI")

# "st.session_state object: ", st.session_state

# if "el_bryan" not in st.session_state:
#     st.session_state["el_bryan"] = "chingon"

# if "el_gordo" not in st.session_state:
#     st.session_state.el_gordo = False

# st.write("Here it:")
# st.write(st.session_state)

# st.write("el_bryan", st.session_state.el_bryan)

# button = st.button("Update state")
# "before pressing", st.session_state

# if button:
#     st.session_state["el_bryan"] = "not chingon"
#     st.session_state.el_gordo+=1 
#     "after pression button", st.session_state
##################################################################################################################################

# Defining Streamlit's session state for state variables
def init():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "run" not in st.session_state:
        st.session_state.run = None

    if "field_ids" not in st.session_state:
        st.session_state.field_ids = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = []


def set_apikey():
    st.sidebar.header("Configure API Key")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password") # password is a valid type of input, the other is "default"

    return api_key

def config(client):
    ais = client.beta.assistants.list(order="desc")
    ais_data = ais.data

    ais_dict = {"Create Assistant" : "create-assistant"}
    for ai in ais_data:
        ais_dict[ai.name] = ai.id
    print(ais_dict)

    # User then select an ai option
    assistant_option = st.sidebar.selectbox("Select Assistant", options= list(ais_dict.keys()))
    
    # This config returns the AI id
    return ais_dict[assistant_option]

def create_new_assistant(client, personal_ai_name, personal_ai_instructions, model_option):
    new_ai = client.beta.assistants.create(
        name=personal_ai_name,
        instructions = personal_ai_instructions,
        model = model_option,
        tools = [
            {
                "type":"code_interpreter"
            }
        ]
    )

def create_assistant(client):
    # Functions to create a personally instructed AI assistant based on user's instructions
    # Will only create it if assistant name not found in overall client assistant's name

    st.divider()
    
    ais_dict = {"Create Assistant" : "create-assistant"}
    personal_ai_name = st.text_input("Name")
    personal_ai_instructions = st.text_area("Instructions")

    # Model list at: https://platform.openai.com/docs/models/overview
    model_option = st.radio("Model", ("gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"))

    # Create assistant if it doesn't exist in client list
    client_ais_dict = dict([(i.name, i.id) for i in client.beta.assistants.list().data])
    if personal_ai_name not in client_ais_dict:
        new_ai = st.button("Create New Bryan AI", 
                           on_click = create_new_assistant, 
                           args=(client, personal_ai_name, personal_ai_instructions, model_option))
        
        # Check that it was created correctly (and is present in client list)
        if new_ai:
            st.write("New personal AI created")

            client_ais_dict = dict([(i.name, i.id) for i in client.beta.assistants.list().data])
            st.success("New personal AI success")
            st.stop()

            print(client_ais_dict)

            return client_ais_dict[personal_ai_name]
        
    # If name already exist in original list, warn user
    else:
        st.warning("AI name already exists, try another one")
            


def main():
    st.title("El Bryan - AI Data Analyst Assistant")
    st.caption("AI Assistant using OpenAI's API")
    st.divider()

    api_key = set_apikey()
    
    if api_key:
        client = OpenAI(api_key=api_key)

        ai_id_option = config(client)
        
        # Work on creating an AI assistant only if first option "create-assistant" is available
        if ai_id_option == "create-assistant":
            print("Creating AI assistant")
            with st.sidebar:
                ai_option = create_assistant(client)
                print(ai_option)
        else:
            st.write("other")

if __name__ == "__main__":
    main()
    # st.write(st.session_state)