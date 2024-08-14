import ollama
import chromadb
import uuid

# initiate the vectordb client
client = chromadb.Client()


# create a vector database
def create_vectordb():
    # delete if the collection exists - during devlopment
    try:
        client.delete_collection(name="conv_collection")
    except:
        pass

    # create collection
    collection = client.create_collection(name="conv_collection")

    # adding a simple conversation
    convsersation = "Prompt: \nWho are you? \nResopnse: \nI'm LlaMa, an AI Assistant"
    conv_embeddings = ollama.embeddings(model="nomic-embed-text", prompt=convsersation)[
        "embedding"
    ]

    collection.add(
        ids=[str(uuid.uuid4())], embeddings=[conv_embeddings], documents=[convsersation]
    )

    return collection


# creating conversation embeddings and save them in vectordb
# each conversation is stored as an embedding
def create_and_save_conv_embeddings(prompt, response, collection):

    # get conversation embeddings
    convsersation = f"Question: {prompt} \nYour Response: {response}"
    conv_embeddings = ollama.embeddings(model="nomic-embed-text", prompt=convsersation)[
        "embedding"
    ]

    collection.add(
        [str(uuid.uuid4())], embeddings=[conv_embeddings], documents=[convsersation]
    )


# function to retrieve semantically similar conversation
def retrieve_similar_conversation(prompt, collection):

    prompt_embeddings = ollama.embeddings(model="nomic-embed-text", prompt=prompt)[
        "embedding"
    ]
    similar_embeddings = collection.query(
        query_embeddings=[prompt_embeddings], n_results=1
    )
    similar_conv = similar_embeddings["documents"][0][0]

    return similar_conv


vector_db = create_vectordb()


while True:
    prompt = input("USER: \n")
    context = retrieve_similar_conversation(prompt=prompt, collection=vector_db)

    final_prompt = f"Instruction: If the answer to the user's question is present in the context, use only the context to answer it \n\nQuestion:\n{prompt} \n\nPrevious Conversation Context: \n{context}"

    print("\n---------------------------Retrieved Context--------------------------")
    print(final_prompt)
    print("---------------------------Retrieved Context--------------------------\n")

    ## Non Streaming Response
    output = ollama.chat(
        model="llama3", messages=[{"role": "user", "content": final_prompt}]
    )
    response = output["message"]["content"]

    create_and_save_conv_embeddings(
        prompt=prompt, response=response, collection=vector_db
    )

    print(f"\nASSISTANT:\n{response}\n")

    ## Streaming response
    # output = ollama.chat(model="llama3", messages=conv_history, stream=True)
    # response = ""

    # print("\nASSISTANT:")
    # for chunk in output:
    #     content = chunk["message"]["content"]
    #     response += content
    #     print(content, end="", flush=True)
    # print("\n")
    # conv_history.append({"role": "assistant", "content": response})
