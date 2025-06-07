# -*- coding: windows-1252 -*-
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def main():
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    # model = init_chat_model("claude-sonnet-4-20250514", model_provider="anthropic")

    messages = [
        SystemMessage("Translate the following from Polish to English"),
        HumanMessage("""Panno œwiêta, co Jasnej bronisz Czêstochowy
I w Ostrej œwiecisz Bramie! Ty, co gród zamkowy
Nowogródzki ochraniasz z jego wiernym ludem!
Jak mnie dziecko do zdrowia powróci³aœ cudem
(Gdy od p³acz¹cej matki, pod Twoj¹ opiekê
Ofiarowany, martw¹ podnios³em powiekê;
I zaraz mog³em pieszo, do Twych œwi¹tyñ progu
Iœæ za wrócone ¿ycie podziêkowaæ Bogu),
Tak nas powrócisz cudem na Ojczyzny ³ono. 
Tymczasem przenoœ moj¹ duszê utêsknion¹
Do tych pagórków leœnych, do tych ³¹k zielonych,
Szeroko nad b³êkitnym Niemnem rozci¹gnionych;
Do tych pól malowanych zbo¿em rozmaitem,
Wyz³acanych pszenic¹, posrebrzanych ¿ytem;
Gdzie bursztynowy œwierzop, gryka jak œnieg bia³a,
Gdzie panieñskim rumieñcem dziêcielina pa³a,
A wszystko przepasane jakby wstêg¹, miedz¹
Zielon¹, na niej z rzadka ciche grusze siedz¹. 
Œród takich pól przed laty, nad brzegiem ruczaju,
Na pagórku niewielkim, we brzozowym gaju,
Sta³ dwór szlachecki, z drzewa, lecz podmurowany;
Œwieci³y siê z daleka pobielane œciany,
Tym bielsze, ¿e odbite od ciemnej zieleni
Topoli, co go broni¹ od wiatrów jesieni. 
Dom mieszkalny niewielki, lecz zewsz¹d chêdogi,
I stodo³ê mia³ wielk¹, i przy niej trzy stogi
U¿¹tku, co pod strzech¹ zmieœciæ siê nie mo¿e.
Widaæ, ¿e okolica obfita we zbo¿e,
I widaæ z liczby kopic, co wzd³u¿ i wszerz smugów 
Œwiec¹ gêsto jak gwiazdy, widaæ z liczby p³ugów
Orz¹cych wczeœnie ³any ogromne ugoru,
Czarnoziemne, zapewne nale¿ne do dworu,
Uprawne dobrze na kszta³t ogrodowych grz¹dek:
¯e w tym domu dostatek mieszka i porz¹dek.
Brama na wci¹¿ otwarta przechodniom og³asza,
¯e goœcinna, i wszystkich w goœcinê zaprasza.""")
    ]

    for token in model.stream(messages):
        print(token.content, end="")

    # response = model.invoke(messages)
    # print(response.content)



if __name__ == "__main__":
    main()
