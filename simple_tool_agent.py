# Import relevant functionality
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# Create the agent
memory = MemorySaver()
model = init_chat_model("gpt-4o-mini", model_provider="openai")
search = TavilySearch(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="Hi im Roman! and i live in Poland, Wroclaw")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


for step in agent_executor.stream(
    {"messages": [HumanMessage(content="Whats the weather where I live today 14.06.2025?")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# for step, metadata in agent_executor.stream(
#     {"messages": [HumanMessage(content="Whats the weather where I live today 14.06.2025?")]},
#     config,
#     stream_mode="messages",
# ):
#     if metadata["langgraph_node"] == "agent" and (text := step.text()):
#         print(text, end="")