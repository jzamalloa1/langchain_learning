from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import streamlit as st
import pandas as pd
import numpy as np
import os, tempfile

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

    if "file_ids" not in st.session_state:
        st.session_state.file_ids = []

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

def delete_file(client, assistant_id, file_id):
    # Deletes files from client
    client.beta.assistants.files.delete(
        assistant_id=assistant_id,
        file_id=file_id
    )

def assistant_handler(client, assistant_id):
    # Executes tasks on created assistants - WILL REDRAW SIDEBAR PORTION CREATED BY FIRST "IF STATEMENT" (see below)
    # This is logical since this function is called as if the "create-assistant" option was never executed, and therefore
    #  the create_assistant() sidebar was never drawn

    ai = client.beta.assistants.retrieve(assistant_id)

    # NOTE: Redrawing sidebar when assistant handler is called. Using ai's original values (as it should be)
    with st.sidebar:
        personal_ai_name = st.text_input("Name", value=ai.name)
        personal_ai_instructions = st.text_area("Instructions", value=ai.instructions)
        model_option = st.radio("Model", ("gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "nagging"))

        # Option to upload files - This grid won't show up unless we upload files
        st.subheader("Manage files")
        grid = st.columns(2)

        # Populate grid with file to handle them
        for file_id in ai.file_ids:
            with grid[0]:
                st.text(file_id)

            # Second column will allow user to delete files
            with grid[1]:
                st.button("Delete", 
                          on_click=delete_file, 
                          key=file_id,
                          args=(client, assistant_id, file_id))
        
        # Add uploader
        uploaded_file = st.file_uploader("Upload File", type=["txt", "csv", "pdf"])

        # Make button to update API assistant based on file input - and new instructions (this seemes redundant)
        # NOTE: This seems redundant, rather than coming from new text input, they should come from previously filled
        #  name, instructions and model option (all part of the client's stored data)
        if st.button("Update Assistant"):
            ai = client.beta.assistants.update(
                assistant_id = assistant_id,
                name = personal_ai_name,
                instructions = personal_ai_instructions,
                model=model_option
            )

            # Add file if it exists to OpenAI and get its id in the client side of OpenAI (created function does this)
            if uploaded_file is not None:
                new_file_id = upload_file(client, assistant_id, uploaded_file)

                # Append file id to session state
                st.session_state.file_ids.append(new_file_id)

                # Notify users
                st.write("File uploaded succesfully")

    # Return updated ai, model option and ai instrunctions
    return ai, model_option, personal_ai_instructions

def upload_file(client, assistant_id, uploaded_file):
    # Function to upload file to OpenAI and get its ID from client's side in OpenAI

    # NOTE: Usage of tempfile: https://docs.python.org/3/library/tempfile.html
    # "Return a file-like object that can be used as a temporary storage area. The file is created securely, 
    #  using the same rules as mkstemp(). It will be destroyed as soon as it is closed (including an implicit 
    #  close when the object is garbage collected). Under Unix, the directory entry for the file is either not 
    #  created at all or is removed immediately after the file is created"
    # "This file-like object can be used in a with statement, just like a normal file." <--

    # Open file and create it with a name
    # delete=False argument will allow the file to exist after closing it
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()

        # Upload file to OpenAI (not added to our assistant, yet)
        # NOTE: Usage of "files.create": https://platform.openai.com/docs/assistants/how-it-works/creating-assistants
        with open(tmp_file.name, "rb") as f:
            response = client.files.create(
                file=f,
                purpose="assistants"
            )
            print(response)

            # Remove file after adding it to clien
            os.remove(tmp_file.name)

    # Attach file id to assistant id so that you can call the file id from the assistant
    ai_file = client.beta.assistants.files.create(
        assistant_id=assistant_id,
        file_id=response.id
    )

    return ai_file.id

def create_new_assistant(client, personal_ai_name, personal_ai_instructions, model_option):
    # Create new AI assistant based on user's input. 
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

        # Returns the AI id and initiates AI list in dropdown menu, with "create-assistant" initially selected
        ai_id_option = config(client)
        
        # This will be triggered first since by default "create-assistant" will be presented as the first available option when
        #  config() is ran. As long as it is not changed, the first if statement will be executed
        if ai_id_option == "create-assistant":
            print("Creating AI assistant")
            with st.sidebar:
                ai_option = create_assistant(client)
                print(ai_option)
        
        # Once an assistant has been created or another option besides "create-assistant" is created, then the assistant-handler
        #  will get to work
        else:
            st.write("Using created assistant")

            # Return update AI with options, file in the system and instructions
            st.session_state.current_assistant, st.session_state.model_option, st.session_state.assistant_instructions = assistant_handler(client, ai_id_option)
            
            # Instantiate thread ID (if it doesn't exist):
            # NOTE: Context on threads: https://platform.openai.com/docs/assistants/how-it-works/objects
            # "Thread: A conversation session between an Assistant and a user. Threads store Messages and 
            #  automatically handle truncation to fit content into a model’s context."
            # You store messages in a Thread
            if st.session_state.thread_id is None:
                st.session_state.thread_id = client.beta.threads.create().id
                print("Thread created", st.session_state.thread_id)
            
            # Create chat prompt function to call chat
            chat_prompt(client, ai_id_option)

def chat_prompt(client, assistant_option):

    # Make placeholde
    # NOTE: On streamlit empty: https://docs.streamlit.io/library/api-reference/layout/st.empty
    placeholder = st.empty()

    # If prompt is added, then modify contents of placeholder to add prompts
    if prompt := st.chat_input("Enter your message here"):
        with placeholder.container():
            with st.chat_message("user"):
                st.markdown(prompt)

        # Place message in session from threads, from clien'ts API call
        st.session_state.messages = client.beta.threads.messages.create(
            thread_id = st.session_state.thread_id,
            role="user",
            content=prompt
        )

        # Then update assistant with session's state variables
        st.session_state.current_assistant = client.beta.assistants.update(
            st.session_state.current_assistant.id,
            instructions = st.session_state.assistant_instructions,
            name = st.session_state.current_assistant.name,
            tools = st.session_state.current_assistant.tools,
            model = st.session_state.model_option,
            file_ids = st.session_state.file_ids
        )

        # Then add the run
        st.session_state.run = client.beta.threads.run.create(
            thread_id = st.session_state.thread_id,
            assistant_id = assistant_option,
            tools = [
                {
                    "type":"code_interpreter"
                }
            ]
        )

        print(st.session_state.run)
        pending=False

if __name__ == "__main__":

    # Initialize session state variables
    init()

    # Call main function
    main()
    # st.write(st.session_state)