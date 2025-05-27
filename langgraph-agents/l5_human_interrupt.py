#!/usr/bin/env python
# coding: utf-8
from langgraph.checkpoint.sqlite import SqliteSaver

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

from uuid import uuid4
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage

"""
In previous examples we've annotated the `messages` state key
with the default `operator.add` or `+` reducer, which always
appends new messages to the end of the existing messages array.

Now, to support replacing existing messages, we annotate the
`messages` key with a customer reducer function, which replaces
messages with the same `id`, and appends them otherwise.
"""
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


tool = TavilySearchResults(tavily_api_key=properties.getTavilyApiKey(), max_results=2)


# ## Manual human approval

class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        logger.debug(state)
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

def print_action(state):
    print(f"actions: {state.values['messages'][-1].tool_calls}")



prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(api_key=properties.getApiKey(), model="gpt-4o-mini")

with  SqliteSaver.from_conn_string(":memory:") as memory:
    abot = Agent(model, [tool], system=prompt, checkpointer=memory)

    messages = [HumanMessage(content="What is the weather in Boston, MA?")]
    thread = {"configurable": {"thread_id": "1"}}
    msgEnv = {"messages": messages}
    while True:
        for event in abot.graph.stream( msgEnv, thread):
         for v in event.values():
            print(v)
            print("---")
        nextState = abot.graph.get_state(thread).next
        print(f"next state: {nextState}")
        if len(nextState) == 0:
            break
        while nextState:
            #print("\n", abot.graph.get_state(thread), "\n")
            print_action(abot.graph.get_state(thread))
            _input = input("proceed?")
            if _input != "y":
                print("aborting")
                exit(0)
            break
        msgEnv = None
