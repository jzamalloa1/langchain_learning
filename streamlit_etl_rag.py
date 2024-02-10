import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langsmith import Client
from langchain import hub
import os

# Trace in LangSmith - Unquote if you are interesting tracking through Langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""
# client = Client()

# Display title
st.title('🦜🔗 ETL RAG App')

# Supporting Functions
def format_docs(docs):
    return "\n\n".join(i.page_content for i in docs)

# GenAI Functions
def etl_doc_retriever(api_key):
    # Builds Vector DB retriever using OMOP CDM ETL Doc

    # Load ETL document
    loader = UnstructuredWordDocumentLoader("sample_docs/WebMD_PBM_ETL_5.0.1_20170606.docx", mode="elements")
    docs = loader.load()

    # Split doc into chunks
    docs_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    
    # Build Vector DB and retriever
    vector_docs_db = FAISS.from_documents(docs_split, OpenAIEmbeddings(api_key=api_key))
    docs_retriever = vector_docs_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Return retriever
    return docs_retriever

def rag_etl(api_key, user_input, retriever, prompt):
    # Function to build RAG for ETL Vector DB retriever

    # Get LLM setup
    openai_model_name = "gpt-4-0125-preview"
    llm = ChatOpenAI(model=openai_model_name, temperature=0.5, openai_api_key=api_key)

    # Build RAG Chain
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()} |  #Noticeable RunnableLambda() wrapping is not needed
        prompt |
        llm |
        StrOutputParser()
    )

    # Stream output
    st.info(
        rag_chain.invoke(user_input)
    )

# Main Function
def main():

    # Ask for OpenAI key on left column
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

    # Ensure correct OpenAI API Key is entered
    if openai_api_key.startswith("sk-"):

        # Get prompt and retriever started so that it's ready to go when user  hits Submit
        prompt    = hub.pull("rlm/rag-prompt")
        retriever = etl_doc_retriever(openai_api_key)

        # Create form to input query
        with st.form("rag_form"):

            # Ask user to enter ETL related question
            query = st.text_area("OMOP ETL Related question:", "how can I query condition_occurrence?")
            # Sample question 2: "what is the new name for the column named "source_concept_id" in the condition_occurrence table?"

            # Creqte submit button
            submitted = st.form_submit_button("Submit Q")

            # Call RAG if correct OpenAI API key is being used
            if submitted:
                rag_etl(openai_api_key, query, retriever, prompt)

    else:
        st.warning('Please enter a valid OpenAI API key')
                
if __name__ == "__main__":

    # Call the main function
    main()