#!/usr/bin/env python

import ssl

from dotenv import load_dotenv
_ = load_dotenv()

# Should fix this by setting up the trust store
ssl._create_default_https_context = ssl._create_unverified_context

from utils import get_doc_tools

vector_tool, summary_tool = get_doc_tools("Oracle-2025-10-K.pdf", "oracle-2025-10k")

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow


llm = OpenAI(model="gpt-4o-mini", temperature=0)

# make your single function-calling agent
agent = FunctionAgent(
    llm=llm,
    tools=[vector_tool, summary_tool],
    verbose=True,
    name="assistant",  # any unique name
)

# wrap it in a workflow (replacement for AgentRunner)
workflow = AgentWorkflow(agents=[agent], root_agent=agent.name)

import asyncio

async def main():
    # send a single user message to the agent
    resp = await workflow.run(user_msg="Does it have a breakout of Global Industry Unit LOB? If it does, what is its contribution to the revenue?")
    print(str(resp))   # stringify the response

asyncio.run(main())
