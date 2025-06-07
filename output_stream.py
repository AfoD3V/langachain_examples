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
                    Wiosna w kwietniu zbudzi�a si� z rana,
                    Wysz�a wprawdzie troszeczk� zaspana,
                    Lecz zajrza�a we wszystkie zak�tki:
                    - Zaczynamy wiosenne porz�dki.

                    Skoczy� wietrzyk zamaszy�cie,
                    Pookurza� mchy i li�cie.
                    Z bocznych dr�ek, z polnych �cie�ek
                    Powymiata� brudny �nie�ek.

                    Krasnoludki wiadra nios�,
                    Myj� ziemi� rann� ros�.
                    Chmury, p�yn�c po b��kicie,
                    Urz�dzi�y wielkie mycie,
                    A ob�oki mi�kk� szmatk�
                    Poleruj� s�o�ce g�adko,
                    A� si� dziwi� wszystkie dzieci,
                    �e tak w niebie �adnie �wieci.
                    Bocian w g�r� poszybowa�,
                    T�cz� barwnie wymalowa�,
                    A �urawie i skowronki
                    Posypa�y kwieciem ��ki,
                    Posypa�y klomby, grz�dki
                    I sko�czy�y si� porz�dki.
                    """
        ),
    ]

    # Stream output
    for token in model.stream(messages):
        print(token.content, end="")


if __name__ == "__main__":
    main()
