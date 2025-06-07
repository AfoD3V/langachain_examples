# -*- coding: windows-1252 -*-
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate

load_dotenv()


def main():
    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    system_template = "Translate the following from Polish into {language}"

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    lang_lst = ["Italian", "Japan", "German"]
    
    for lang in lang_lst:
        prompt = prompt_template.invoke({"language": lang, "text": "Czeœæ! Jak siê masz?"})
        response = model.invoke(prompt)
        print(response.content)


if __name__ == "__main__":
    main()
