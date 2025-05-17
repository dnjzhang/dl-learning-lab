#!/usr/bin/env python
# coding: utf-8

# == common code block
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

# == end of common code block

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.tools import Tool
from tavily import TavilyClient



class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:

    def __init__(self, model, tools, system=""): # create graph, compile graph, bind model to tools
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            logger.debug(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                logger.error("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        logger.debug("Back to the model!")
        return {'messages': results}

import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Query LLM with the help of a search engine."
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        help="The search query string to process."
    )
    args = parser.parse_args()

    # Now you can use args.query in your code
    query = args.query

    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """

    model = ChatOpenAI(api_key=properties.getApiKey(), model="gpt-4o-mini")

    # Initialize TavilyClient
    tavily_tool = TavilySearchResults(tavily_api_key=properties.getTavilyApiKey())

    abot = Agent(model, [tavily_tool], system=prompt)

    messages = [HumanMessage(content=query)]
    result = abot.graph.invoke({"messages": messages})

    print(f"results: {result['messages'][-1].content}")

if __name__ == "__main__":
    main()

