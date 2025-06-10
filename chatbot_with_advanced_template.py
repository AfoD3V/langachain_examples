from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

load_dotenv()
model = init_chat_model("gpt-4o-mini", model_provider="openai")

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Class storing state
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


# Define the function that calls the model
def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}


def main():
    # Define a new graph
    workflow = StateGraph(state_schema=State)

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # Set config
    config = {"configurable": {"thread_id": "abc123"}}

    # Set Language
    language = ""

    while True:
        if not language:
            language = input("Language preference: ")
            query = input("You: ")
            input_messages = [HumanMessage(query)]
            output = app.invoke({"messages": input_messages, "language": language}, config)
        else:
            query = input("You: ")
            if query.lower() == "quit":
                break
            input_messages = [HumanMessage(query)]
            output = app.invoke({"messages": input_messages}, config)

        output["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
