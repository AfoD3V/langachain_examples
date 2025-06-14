import asyncio
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()
model = init_chat_model("gpt-4o-mini", model_provider="openai")


# Define the function that calls the model
async def call_model(state: MessagesState):
    response = await model.ainvoke(state["messages"])
    return {"messages": response}


async def main():
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
            output = await app.ainvoke({"messages": input_messages}, config)
            output["messages"][
                -1
            ].pretty_print()  # output contains all messages in state


if __name__ == "__main__":
    asyncio.run(main())
