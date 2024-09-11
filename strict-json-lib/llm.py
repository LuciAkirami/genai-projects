import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")


def llm(system_prompt: str, user_prompt: str) -> str:
    """Here, we use OpenAI for illustration, you can change it to your own LLM"""
    # ensure your LLM imports are all within this function

    # define your own LLM here
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = system_prompt + "\n\n" + user_prompt
    # print(prompt)
    chat = model.start_chat(history=[])

    response = chat.send_message(prompt)
    # print(response.text)
    return response.text
