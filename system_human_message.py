# -*- coding: windows-1252 -*-
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def main():
    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    messages = [
        SystemMessage("Translate the following from Polish to English"),
        HumanMessage("Hej jak siê masz?")
    ]

    response = model.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    main()
