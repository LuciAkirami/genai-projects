import ollama
import chromadb
import psycopg
from psycopg.rows import dict_row
from rich import print
import uuid

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
conv = []

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
def retrieve_conversations(prompt, n):
    query_embeddings = ollama.embeddings(model="nomic-embed-text", prompt=prompt)[
        "embedding"
    ]
    similar_conv = vectordb.query(query_embeddings=[query_embeddings], n_results=n)[
        "documents"
    ][0]

    # the retrieve more than 1 result, concat them with a breakline
    if n > 1:
        similar_conv = "\n\n".join(similar_conv)
        return similar_conv

    return similar_conv[0]


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


# fetch the conversations and create a vectordb out of it
conversations = fetch_conversations()
vectordb = create_vectordb(conversations=conversations)

# testing retrieve conversations
# print(retrieve_conversations("Paris in winter", 1))


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
    relevant_conv = retrieve_conversations(user_prompt, 1)
    final_prompt = f"Sytem Prompt: \n{system_prompt} \n\nQuestion: \n{user_prompt} \n\nPrevious Memory Context: \n{relevant_conv} \nEND of Previous Memory Context"

    # to check the if the final prompt is containing the relevant messages
    # print(f"Final Prompt: \n{final_prompt}")

    # append the final_prompt to the list of conversations
    conv.append({"role": "user", "content": final_prompt})
    response = streaming_response(conv)
    # append the response to the list of conversations
    conv.append({"role": "assistant", "content": response})

    # store the conversation in the sql db
    save_conversation(user_prompt, response)

    # store the conversation in the chromadb
    save_conv_vectordb(user_prompt, response)
