import chromadb
import ollama
import uuid
from tqdm import tqdm

# instantiate a chroma client
chroma_client = chromadb.Client()


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
def retrieve_conversations(vectordb, query_list, n):
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
                if "yes" in is_retrieved_conversation_relevant(query, conversation):
                    similar_documents.append(conversation)

    similar_documents = "\n\n".join(similar_documents)

    return similar_documents


# save the conversation in the vector database
def save_conv_vectordb(vectordb, prompt, response):
    prompt_response_pair = f"Prompt: \n{prompt} \nResponse: \n{response}"
    conv_embeddings = ollama.embeddings(
        model="nomic-embed-text", prompt=prompt_response_pair
    )["embedding"]

    vectordb.add(
        ids=[str(uuid.uuid4())],
        embeddings=[conv_embeddings],
        documents=[prompt_response_pair],
    )


# checking if retrieved conversation is correct or not
def is_retrieved_conversation_relevant(query, conversation):
    relevance_check_msg = (
        "You are an AI designed to determine the relevance of a provided context to a user's query. "
        "You will receive a user query that includes a context retrieved from conversation history. "
        "Your task is to analyze whether the provided context is directly relevant and useful in answering the user's query. "
        "If the context is relevant to the query, respond with 'yes'. If the context is not relevant, respond with 'no'. "
        "Provide only 'yes' or 'no' as your response, without any explanations."
    )

    relevance_convo = [
        {"role": "system", "content": relevance_check_msg},
        {
            "role": "user",
            "content": f"Query: {query} Retrieved Context: {conversation}",
        },
    ]

    response = ollama.chat(model="llama3", messages=relevance_convo)["message"][
        "content"
    ]

    return response
