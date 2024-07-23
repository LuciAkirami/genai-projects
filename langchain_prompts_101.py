from langchain_core.prompts import (
    PromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rich import print as rprint

# ------ Simple Prompt ---------
prompt = PromptTemplate.from_template("Tell me a joke on {topic}")
rprint(prompt.format(topic="Car"), end="\n\n")
# Output: Tell me a joke on Car

prompt_ac = prompt.invoke({"topic": "Panther"})
rprint(prompt_ac, "\n")
# Output: StringPromptValue(text='Tell me a joke on Panther')

rprint(prompt_ac.to_string(), "\n")
# Output: Tell me a joke on Panther

rprint(prompt_ac.to_messages(), "\n")
# Output: [HumanMessage(content='Tell me a joke on Panther')]

# -------- Chat Prompt ---------
chat_prompt = ChatPromptTemplate.from_messages(
    [("human", "Hello, how are you?"), ("ai", "I'm doing well, thanks!")]
)
rprint(chat_prompt, "\n")
# Output:
# ChatPromptTemplate(
#     input_variables=[],
#     messages=[
#         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='Hello, how are you?')),
#         AIMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template="I'm doing well, thanks!"))
#     ]
# )
# [SystemMessage(content='You are an AI assistant.'), HumanMessage(content='Hello!')]


# ------ MessagePlaceholder -------
# Here, we can use message placeholder so that, we can replace it with some other messages
# in the future
# Here, we create a simple prompt with a MessagePlaceholder, hence no message, and provide it the name history
prompt = MessagesPlaceholder("history", optional=True)
rprint(prompt.format_messages(), "\n")  # returns empty list []

rprint(
    # now during format, we populate our placeholder with the actual list of messages
    prompt.format_messages(
        history=[
            ("system", "You are an AI assistant."),
            ("human", "Hello!"),
        ]
    ),
    "\n",
)

# another example
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        # then we create a placeholder
        # optional=True is necessary, which will remove place holder if not used
        # if we do not set this and call .format_message() without history, it gives "KeyError"
        MessagesPlaceholder("history", optional=True),
        ("human", "What are blackholes?"),
    ]
)

# here, as we are not providing any history, the optional=True will remove the placeholder
# and the propmt will contain only system and human message
rprint(prompt.format_messages(), "\n")
# Output:
# [SystemMessage(content='You are a helpful chatbot'), HumanMessage(content='What are blackholes?')]

rprint(
    prompt.format_messages(
        history=[
            ("human", "What is the plural word for fish?"),
            ("ai", "The plural word for fish is fishes"),
        ]
    ),
    "\n",
)
# Output
# [
#     SystemMessage(content='You are a helpful chatbot'),
#     HumanMessage(content='What is the plural word for fish?'),
#     AIMessage(content='The plural word for fish is fishes'),
#     HumanMessage(content='What are blackholes?')
# ]


# Example with Prompt Invoke
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful chatbot"
        ),  # or ('system','You are a helpful chatbot')
        # then we create a placeholder
        # optional=True is necessary, which will remove place holder if not used
        # if we do not set this and call .format_message() without history, it gives "KeyError"
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ]
)

# here, as we are not providing any history, the optional=True will remove the placeholder
# and the propmt will contain only system and human message
rprint(prompt.format_messages(question="How are you?"), "\n")
# Output:
# [SystemMessage(content='You are a helpful chatbot'), HumanMessage(content='How are you?')]

rprint(
    prompt.invoke(
        {
            "history": [
                HumanMessage(content="What is the plural word for fish?"),
                AIMessage(content="The plural word for fish is fishes"),
            ],
            "question": "Summarize above conversation",
        }
    ),
    "\n",
)
# Output
# ChatPromptValue(
#     messages=[
#         SystemMessage(content='You are a helpful chatbot'),
#         HumanMessage(content='What is the plural word for fish?'),
#         AIMessage(content='The plural word for fish is fishes'),
#         HumanMessage(content='What is Machine Learning?')
#     ]
# )

# ------- Converting the Prompts to String --------
prompt_ac = prompt.invoke(
    {
        "history": [
            HumanMessage(content="What is the plural word for fish?"),
            AIMessage(content="The plural word for fish is fishes"),
        ],
        "question": "What is Machine Learning?",
    }
)

rprint(prompt_ac.to_string(), "\n")
# Output:
# System: You are a helpful chatbot
# Human: What is the plural word for fish?
# AI: The plural word for fish is fishes
# Human: What is Machine Learning?

# -------- Pretty Printing Prompts ----------
prompt.pretty_print()
# Output:
# ================================ System Message ================================

# You are a helpful chatbot

# ============================= Messages Placeholder =============================

# {history}

# ================================ Human Message =================================

# {question}
