# -*- coding: windows-1252 -*-
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


def main():
    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    messages = [
        SystemMessage("Translate the following from Polish to English"),
        HumanMessage(
            """
                    Wiosna w kwietniu zbudzi³a siê z rana,
                    Wysz³a wprawdzie troszeczkê zaspana,
                    Lecz zajrza³a we wszystkie zak¹tki:
                    - Zaczynamy wiosenne porz¹dki.

                    Skoczy³ wietrzyk zamaszyœcie,
                    Pookurza³ mchy i liœcie.
                    Z bocznych dró¿ek, z polnych œcie¿ek
                    Powymiata³ brudny œnie¿ek.

                    Krasnoludki wiadra nios¹,
                    Myj¹ ziemiê rann¹ ros¹.
                    Chmury, p³yn¹c po b³êkicie,
                    Urz¹dzi³y wielkie mycie,
                    A ob³oki miêkk¹ szmatk¹
                    Poleruj¹ s³oñce g³adko,
                    A¿ siê dziwi¹ wszystkie dzieci,
                    ¯e tak w niebie ³adnie œwieci.
                    Bocian w górê poszybowa³,
                    Têczê barwnie wymalowa³,
                    A ¿urawie i skowronki
                    Posypa³y kwieciem ³¹ki,
                    Posypa³y klomby, grz¹dki
                    I skoñczy³y siê porz¹dki.
                    """
        ),
    ]

    # Stream output
    for token in model.stream(messages):
        print(token.content, end="")


if __name__ == "__main__":
    main()
