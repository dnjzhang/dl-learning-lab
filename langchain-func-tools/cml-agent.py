#!/usr/bin/env python
# coding: utf-8
# TODO: add logging and add command line argument
import argparse
import json
import warnings
# # Command Line Conversational agent
from datetime import datetime, timezone

import requests
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.tools import tool
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import format_tool_to_openai_function, convert_to_openai_function
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from wikipedia import PageError, DisambiguationError

from utils.load_properties import LoadProperties

properties = LoadProperties()
api_key = properties.getApiKey()


# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")


@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    #print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))

    current_utc_time = datetime.now(timezone.utc)
    #print("Current UTC time:", current_utc_time)
    time_list = [
        # 1) replace the “Z” with “+00:00” → fromisoformat yields tz‐aware dt
        # 2) astimezone(timezone.utc) ensures tzinfo is exactly timezone.utc
        datetime.fromisoformat(t.replace("Z", "+00:00"))
                .astimezone(timezone.utc)
        for t in results["hourly"]["time"]
    ]
    temperature_list = results['hourly']['temperature_2m']

    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]

    return f'The current temperature is {current_temperature}°C'


import wikipedia


@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (PageError,
                DisambiguationError,
                ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


tools = [get_current_temperature, search_wikipedia]

from langchain.prompts import ChatPromptTemplate
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant. Use the tools to answer the questions. If the tools are not applicable, answer the question based on your own knowldge."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

functions = [convert_to_openai_function(f) for f in tools]

model = ChatOpenAI(api_key=api_key, temperature=0).bind(functions=functions)

agent_chain = RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt | model | OpenAIFunctionsAgentOutputParser()

warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, memory=memory)


parser = argparse.ArgumentParser(
    description="Chat with LLM with a question, feel free to introduce yourself"
)
parser.add_argument(
    "--question",
    required=True,
    help="The question string to be processed"
)
parser.add_argument(
    "--intro",
    default="john",
    help="Introductory name or phrase (defaults to 'john')"
)

args = parser.parse_args()
question = args.question
intro = args.intro


results = agent_executor.invoke({"input": "My name is " +intro})
print("LLM: "+results["output"])
print(f"You asked: {question}")
results = agent_executor.invoke({"input": question})
print("LLM:" + results["output"])
