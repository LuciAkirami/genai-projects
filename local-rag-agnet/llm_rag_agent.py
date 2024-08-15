from database import fetch_conversations, save_conversation
from vectordb import create_vectordb, retrieve_conversations, save_conv_vectordb
from colorama import Fore
import ollama
import ast

# create a system prompt
system_prompt = (
    "You are an AI assistant with the ability to recall and use previous conversations with the user. "
    "For each user question, you will be given a relevant portion of the conversation history as context. "
    "If the provided context directly answers the user's query, seamlessly integrate that information into your response without any reference to the context or prior conversations. "
    "If the context is not relevant to the current query, ignore it and respond as you normally would. "
    "Do not reference the conversation history, mention that you are using prior context, or use phrases like 'as discussed before' or 'building on our previous conversation.' "
    "Your response should be as if it was generated independently, providing the most accurate and helpful answer based solely on the current query."
)

# create a list for the messages conversation
conv = [{"role": "system", "content": system_prompt}]


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

    print(Fore.LIGHTGREEN_EX + "\nASSISTANT: ")
    for chunk in output:
        content = chunk["message"]["content"]
        response += content
        print(content, end="", flush=True)
    print("\n")

    return response


while True:
    user_prompt = input(Fore.WHITE + "USER: \n")
    if user_prompt[-4:].lower() == "exit":
        break

    query_list = create_queries(user_prompt)
    print(Fore.YELLOW + "Performing Vector Search with following Queries: ", query_list)

    relevant_conv = retrieve_conversations(vectordb, query_list, 2)
    final_prompt = f"Question: \n{user_prompt} \n\nPrevious Memory Context: \n{relevant_conv} \nEND of Previous Memory Context"

    # # to check the if the final prompt is containing the relevant messages
    print(f"Final Prompt: \n{final_prompt}")

    # append the final_prompt to the list of conversations
    conv.append({"role": "user", "content": final_prompt})
    response = streaming_response(conv)
    # append the response to the list of conversations
    conv.append({"role": "assistant", "content": response})

    # store the conversation in the sql db
    save_conversation(user_prompt, response)

    # store the conversation in the chromadb
    save_conv_vectordb(vectordb, user_prompt, response)
