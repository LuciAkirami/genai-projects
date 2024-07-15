import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql.base import SQLDatabaseChain

load_dotenv()

# connecting the ChinookDB
# sample_rows_in_table_info specifies how many samples to show when
# table_info is caled
db = SQLDatabase.from_uri("sqlite:///chinook.db", sample_rows_in_table_info=3)
print("Num Tables:", len(db.get_usable_table_names()))
print("Table Names:\n", db.get_usable_table_names())
print("Table Info:\n", db.table_info)

llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="mixtral-8x7b-32768")

# creating our SQLChain
sql_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
result = sql_chain.run("Which country's customers spent the least?")
print("LLM Result: ", result)

# getting the prompt template of the chain
print("Prompt Template")
print(sql_chain.llm_chain.prompt.template)
