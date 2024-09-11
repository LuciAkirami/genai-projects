from strictjson import *
from llm import llm

res = strict_json(
    system_prompt="You are a classifier",
    user_prompt="The movie was so bad that I slept midway",
    output_format={
        "Sentiment": "Type of Sentiment",
        "Adjectives": "Array of adjectives",
        "Words": "Number of words",
    },
    llm=llm,
)

print(res, end="\n\n")

### Code Generation

res = strict_json(
    system_prompt="You are a code generator, generating code to fulfil a task",
    user_prompt="Given array p, output a function named func_sum to return its sum",
    output_format={"Elaboration": "How you would do it", "C": "Code", "Python": "Code"},
    llm=llm,
)

print(res, end="\n\n")
print(res["Python"])