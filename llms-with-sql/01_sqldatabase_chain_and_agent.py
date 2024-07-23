"""
Best to check the below with closed source model like OpenAI. The open source models may not provide accurate results
"""

import os
from rich import print as rprint
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql.base import SQLDatabaseChain

load_dotenv()

# connecting the ChinookDB
# sample_rows_in_table_info specifies how many samples to show when
# table_info is called
db = SQLDatabase.from_uri("sqlite:///chinook.db", sample_rows_in_table_info=3)
print("Num Tables:", len(db.get_usable_table_names()))
print("Table Names:\n", db.get_usable_table_names())
print("Table Info:\n", db.table_info)

llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="mixtral-8x7b-32768")

# creating our SQLChain
sql_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
result = sql_chain.run("Which country's customers spent the least?")
print("LLM Result: ", result)
# most of the time the llama3-70b fails to answer the question

# getting the prompt template of the chain
print("\n--------Prompt Template--------\n")
print(sql_chain.llm_chain.prompt.template)

# let us try asking two questions to the model
try:
    result = sql_chain.run(
        "Which country's customers spent the second least? And along with that, list the total sales per country for all the countries "
    )
    print("LLM Result: ", result)
except Exception as e:
    print("Model failed to answer")

# ----------------------- SQL Agent -------------------------------------
"""
Advantages of SQL Agents:

It will save tokens, by only retrieving the schema from relevant tables. It can recover from errors by running a 
generated query, catching the traceback and regenerating it correctly. chain runs on predefined pattern while agent is defined 
by llm and llm dynamically do reasoning to find out which tool needs to be run
"""

from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent

llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="llama3-70b-8192")

# use tool-calling for other llms and openai-tools for openai
agent_executor = create_sql_agent(
    llm, db=db, agent_type="tool-calling", verbose=True, stream_runnable=False
)
print("\n------------------SQL AGENT------------------\n")
# result = agent_executor.invoke("Which country's customers spent the second most?")
# print("LLM Result: ", result["output"])

# let us try asking two questions here. The model passes
result = agent_executor.invoke(
    "Which country's customers spent the second least? And along with that, list the total sales per country for all the countries "
)
print("LLM Result: ", result["output"])

# prompt that the agent is using
print("\n------------------- Agent Prompt --------------------\n")
print(agent_executor.agent.runnable.get_prompts()[0].messages[0].content)

# tools used by the agent
print("\n------------------- Agent Tools ---------------------\n")
rprint(agent_executor.agent.runnable.to_json()["kwargs"]["middle"][1].kwargs["tools"])
