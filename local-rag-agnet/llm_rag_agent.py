import ollama
import chromadb
import psycopg
from psycopg.rows import dict_row
import uuid
import ast
from tqdm import tqdm

# DB Parameters
DB_PARAMS = {
    "dbname": "memory_agent",
    "user": "example_user",
    "password": "12345",
    "host": "localhost",
    "port": "5432",
}

# create a system prompt
system_prompt = (
    "You are an AI assistant with the ability to recall and use previous conversations with the user. "
    "For each user question, you will be given a relevant portion of the conversation history as context. "
    "If the provided context directly answers the user's query, incorporate that information seamlessly into your response. "
    "If the context is not relevant to the current query, ignore it and respond as you normally would. "
    "Do not reference the conversation history or mention that you are using it in your response. Simply provide the most accurate and helpful answer, "
    "as if you were responding without access to prior context."
)

# create a list for the messages conversation
conv = [{"role": "system", "content": system_prompt}]

# instantiate a chroma client
chroma_client = chromadb.Client()


# fetch the conversation from the database
def fetch_conversations():
    conn = psycopg.connect(**DB_PARAMS)

    with conn.cursor(row_factory=dict_row) as cursor:
        fetched_data = cursor.execute("SELECT * FROM conversations;")
        conversation = fetched_data.fetchall()

    conn.close()

    return conversation


# save a conversation in the database
def save_conversation(prompt, response):
    conn = psycopg.connect(**DB_PARAMS)

    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            INSERT INTO conversations (timestamp, prompt, response)
            VALUES (CURRENT_TIMESTAMP, %s, %s);
            """,
            (prompt, response),
        )
        conn.commit()
    conn.close()


# create a vector database and store the conversation embeddings
def create_vectordb(conversations):
    # delete if collection already exists
    try:
        chroma_client.delete_collection("conv-collection")
    except Exception as e:
        pass

    vectordb = chroma_client.create_collection("conv-collection")

    for conversation in conversations:
        prompt_response_pair = f"Prompt: \n{conversation['prompt']} \nResponse: \n{conversation['response']}"
        conv_embeddings = ollama.embeddings(
            model="nomic-embed-text", prompt=prompt_response_pair
        )["embedding"]
        vectordb.add(
            ids=[str(uuid.uuid4())],
            embeddings=[conv_embeddings],
            documents=[prompt_response_pair],
        )

    return vectordb


# retrieve the most similar conversations
def retrieve_conversations(query_list, n):
    similar_documents = []

    for query in tqdm(query_list, desc="Extracting Similar Queries"):
        query_embeddings = ollama.embeddings(model="nomic-embed-text", prompt=query)[
            "embedding"
        ]
        similar_conv = vectordb.query(query_embeddings=[query_embeddings], n_results=n)[
            "documents"
        ][0]

        for conversation in similar_conv:
            if conversation not in similar_documents:
                similar_documents.append(conversation)

    similar_documents = "\n\n".join(similar_documents)

    return similar_documents


# save the conversation in the vector database
def save_conv_vectordb(prompt, response):
    prompt_response_pair = f"Prompt: \n{prompt} \nResponse: \n{response}"
    conv_embeddings = ollama.embeddings(
        model="nomic-embed-text", prompt=prompt_response_pair
    )["embedding"]

    vectordb.add(
        ids=[str(uuid.uuid4())],
        embeddings=[conv_embeddings],
        documents=[prompt_response_pair],
    )


# create multiple queries from user's prompt, so to retrieve accurate conversation histories
def create_queries(prompt):
    query_msg = (
        "You are a precise query generation AI. "
        "Your task is to create a Python list of search queries that will be used to search an embedding database of all conversations "
        "you have had with the user. Generate a list of queries that would retrieve the most relevant information necessary to address "
        "the user's current question. Your response must be a Python list, with each query in the form of a string. "
        "Do not explain anything, and ensure the syntax is perfect."
    )

    query_convo = [
        {"role": "system", "content": query_msg},
        {
            "role": "user",
            "content": "What are some good indoor plants that require low maintenance?",
        },
        {
            "role": "assistant",
            "content": '["indoor plants", "low maintenance plants", "easy care houseplants", "best indoor plants"]',
        },
        {
            "role": "user",
            "content": "What are some must-see places in Paris for a first-time visitor?",
        },
        {
            "role": "assistant",
            "content": '["must-see places in Paris", "first-time visitor Paris", "top attractions in Paris", "Paris travel guide"]',
        },
        {"role": "user", "content": "How can I improve my car’s fuel efficiency?"},
        {
            "role": "assistant",
            "content": '["improve car fuel efficiency", "fuel-saving tips", "car maintenance for fuel efficiency", "driving tips to save fuel"]',
        },
        {"role": "user", "content": prompt},
    ]

    query_list = ollama.chat(model="llama3", messages=query_convo)["message"]["content"]

    try:
        # convert the llms string response to an actual python list
        # llm_ouput -> "['hello','hello2']" which is a string as it generates string response not list
        # we need -> ['hello','hello2'] which is a list
        query_list = ast.literal_eval(query_list)
        return query_list
    except:
        return prompt


# fetch the conversations and create a vectordb out of it
conversations = fetch_conversations()
vectordb = create_vectordb(conversations=conversations)


# # streaming response
def streaming_response(prompt):
    output = ollama.chat(model="llama3", messages=prompt, stream=True)
    response = ""

    print("\nASSISTANT: ")
    for chunk in output:
        content = chunk["message"]["content"]
        response += content
        print(content, end="", flush=True)
    print("\n")

    return response


while True:
    user_prompt = input("USER: \n")
    if user_prompt[-4:].lower() == "exit":
        break

    query_list = create_queries(user_prompt)
    print(query_list)

    relevant_conv = retrieve_conversations(query_list, 2)
    final_prompt = f"Question: \n{user_prompt} \n\nPrevious Memory Context: \n{relevant_conv} \nEND of Previous Memory Context"

    # # to check the if the final prompt is containing the relevant messages
    print(f"Final Prompt: \n{final_prompt}")

    # append the final_prompt to the list of conversations
    conv.append({"role": "user", "content": final_prompt})
    response = streaming_response(conv)
    # append the response to the list of conversations
    conv.append({"role": "assistant", "content": response})

    # # store the conversation in the sql db
    # save_conversation(user_prompt, response)

    # # store the conversation in the chromadb
    # save_conv_vectordb(user_prompt, response)
