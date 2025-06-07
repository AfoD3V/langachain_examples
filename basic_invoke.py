# -*- coding: windows-1252 -*-
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

def main():
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    # model = init_chat_model("claude-sonnet-4-20250514", model_provider="anthropic")

    messages = "Hello!"

    response = model.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    main()
