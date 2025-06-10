from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
model = init_chat_model("gpt-4o-mini", model_provider="openai")

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You talk like a pirate. Answer all questions to the best of your ability.",
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}


def main():
    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "abc123"}}

    while True:
        query = input("You: ")
        if query.lower() == "quit":
            break
        else:
            input_messages = [HumanMessage(query)]
            output = app.invoke({"messages": input_messages}, config)
        output["messages"][-1].pretty_print()  # output contains all messages in state


if __name__ == "__main__":
    main()
