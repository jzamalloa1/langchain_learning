# Modified version of ra_pilot.py with Arxiv retrievers

import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import requests
from bs4 import BeautifulSoup
import json
from IPython.display import display, Markdown, Latex
from fastapi import FastAPI
from langserve import add_routes
from langchain.retrievers import ArxivRetriever


# Set environment variable
os.environ["OPENAI_API_KEY"] = ""

# Define custom functions
def scrape_text(url: str):
    # Send a GET request to the webpage
    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"

ddg_search = DuckDuckGoSearchAPIWrapper()
def web_search(query:str, num_results:int = 3):
    results = ddg_search.results(query, num_results)

    return [r["link"] for r in results]

# Fuction to collapse list of lists
def collapse_list_of_lists(lists):
    
    all_text = []
    for i in lists:
        all_text.append("\n\n".join(i))

    return "\n\n".join(all_text)

# Pre-specify Model name
model_name = "gpt-3.5-turbo-16k"

# Define arxiv retriever
retriever = ArxivRetriever(top_k_results=5)

############################################## PROMPTS #############################################

# Define summary template and prompt
summary_template = """{doc}

Using the above text, answer in short the following question

> {question}

---
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats, etc.
"""

summary_prompt = ChatPromptTemplate.from_template(summary_template)

# Define search prompts
search_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI Research assistant. Your name is El Bryan"),
        ("user",
         "Write 3 google search queries to search online that form an objective opinion from "
         "the following {question}\n"
         "You must respond with a list of strings in the following format:"
         '["query 1", "query 2", "query 3"].'
        )
    ]
)

# Prompt to summarize flattened text (summarizer of summaries)
writer_system_prompt = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, "\
"critically acclaimed, objective and structured reports on given text."

research_report_template = """Information:
---------
{research_summary}
---------

Using the above information, answer the following question or topic: {question} in a detailed report. \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.

You should strive to write the report as long as you can using all relevant and necessary information provided. \
You must write the report with markdown syntax. \
Use an unbiased and journalistic tone. \
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions. \
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each. \
You MUST write the report in apa format. \
Cite search results using inline notations. Only cite the most \
relevant results that answer the query accurately. Place these citations at the end \
of the sentence or paragraph that reference them. \
Please do your best, this is very important to my career.
"""

research_summarization_prompt = ChatPromptTemplate.from_messages([
    ("system", writer_system_prompt),
    ("user", research_report_template)
])

############################################################################################################

#################################### CHAINS ################################################################
# Define question to arxiv summarization chain
docs_to_summary_chain = RunnablePassthrough.assign(
    summary_output = summary_prompt | ChatOpenAI(model=model_name) | StrOutputParser()
) | (lambda x: f"Title: {x['doc'].metadata['Title']}\n\n SUMMARY: {x['summary_output']}")

question_doc_chain = RunnablePassthrough.assign(
    docs = lambda x: retriever.get_summaries_as_docs(x["question"])
) | (lambda x: [{"question":x["question"], "doc":u} for u in x["docs"]]) 

question_to_summaries_chain = question_doc_chain | docs_to_summary_chain.map()

# Question formulating chain
search_questions_chain = search_prompt | ChatOpenAI(model=model_name, temperature=0.3) | StrOutputParser() | json.loads | (
    lambda x : [{"question":i} for i in x]
)

# Main research chain mapped over scrapping and summaries
main_research_chain = search_questions_chain | question_to_summaries_chain.map()

# Overall chain 
all_chain = RunnablePassthrough.assign(
    research_summary = main_research_chain | collapse_list_of_lists 
) | research_summarization_prompt | ChatOpenAI(model="gpt-3.5-turbo-16k") | StrOutputParser()

############################################################################################################

########################################## LANGSERVE IT ###################################################

app = FastAPI(
  title="RA Server",
  version="1.0ish",
  description="An Involved Research Assistant Server"
)

add_routes(
    app,
    all_chain,
    path="/El-Bryan",

)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
############################################################################################################