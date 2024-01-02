from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
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