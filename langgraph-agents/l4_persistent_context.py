#!/usr/bin/env python
# coding: utf-8

# Local Environment setup
from utils.load_properties import LoadProperties
properties = LoadProperties()
import logging

# Configure the root logger to INFO level and set a simple format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# End of Local Environment setup

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import json
tool = TavilySearchResults(tavily_api_key=properties.getTavilyApiKey(), max_results=2)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


from langgraph.checkpoint.sqlite import SqliteSaver


class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        self.state = AgentState

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        logger.debug("Sending to LLM: %s",
             [{"content": m.content} for m in messages]
        )
        message = self.model.invoke(messages)
        logger.debug("Received from LLM: %s", message)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            logger.debug(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        logger.debug("Back to the model!")
        return {'messages': results}

def print_message(prefix, message):
    # print(
    #     f"{message.type}: {message.content}"
    #     if isinstance(message, SystemMessage)
    #     else f"{message.type}: {message.content.content}"
    # )
    print(f"{prefix}>> Message type: {type(message)}/{message.type}")
    print(f"\n{prefix}>> Content: {message.content}")
    print(f"\n{prefix}>> Additional Keywords:")
    print(json.dumps(message.additional_kwargs, indent=2))

    # Response Metadata (token usage, model info, etc)
    # print("\n🔍 Response Metadata:")
    #print(json.dumps(message.response_metadata, indent=2))


with SqliteSaver.from_conn_string(":memory:") as memory:

    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    model = ChatOpenAI(api_key=properties.getApiKey(), model="gpt-4o-mini")
    abot = Agent(model, [tool], system=prompt, checkpointer=memory)


    messages = [HumanMessage(content="What is the weather in sf?")]

    thread1 = {"configurable": {"thread_id": "1"}}

    from pprint import pprint

    for event in abot.graph.stream({"messages": messages}, thread1):
        snapshot = abot.graph.get_state(thread1)
        for msg in snapshot.values['messages']:
            print_message("STATE", msg)
        print("==")
        for v in event.values():
            print_message("MSG", v['messages'][-1])
        print("---")
