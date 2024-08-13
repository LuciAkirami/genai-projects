import ollama

# continous conversation
conv_history = []

while True:
    prompt = input("USER: \n")
    conv_history.append({"role": "user", "content": prompt})

    ## Non Streaming Response
    output = ollama.chat(model="llama3", messages=conv_history)
    response = output["message"]["content"]

    print(f"\nASSISTANT:\n{response}\n")
    conv_history.append({"role": "assistant", "content": response})

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

## when the conv exceeds the model's context length, ollama will automatically trim the
## earlier responses
