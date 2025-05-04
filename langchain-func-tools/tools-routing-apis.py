#!/usr/bin/env python
# coding: utf-8
from langchain_openai import ChatOpenAI
from wikipedia import PageError, DisambiguationError

# # Tools and Routing

from utils.load_properties import LoadProperties
properties = LoadProperties()
api_key = properties.getApiKey()

from langchain.agents import tool
from pydantic import BaseModel, Field
import requests
import datetime

import logging

# Configure the root logger to INFO level and set a simple format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")


# Fetch current temperature for given coordinates using the Open-Meteo API
@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> str:
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
    if response.status_code != 200:
        raise Exception(f"API request failed with status code: {response.status_code}")

    results = response.json()

    # Get the current UTC time
    current_utc_time = datetime.datetime.utcnow()

    # Parse the list of time strings into datetime objects
    time_list = [
        datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        for time_str in results['hourly']['time']
    ]
    temperature_list = results['hourly']['temperature_2m']

    # Find the index of the time closest to now
    closest_index = min(
        range(len(time_list)),
        key=lambda i: abs(time_list[i] - current_utc_time)
    )
    current_temperature = temperature_list[closest_index]

    return f'The current temperature is {current_temperature}°C'

# print(f"Get current temperature function description: {get_current_temperature.description}")

#from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_function
#print(f"open ai function: {convert_to_openai_function(get_current_temperature)}")

#print(f"Get sample current temperature:" + get_current_temperature({"latitude": 13, "longitude": 14}))
#print("=========\n")

import wikipedia
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    logger.debug(f"Wiki searching key: {query}")
    page_titles = wikipedia.search(query)
    summaries = []

    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except ( PageError, DisambiguationError ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)

#print(f"Search wikipedia sample=: {search_wikipedia({"query": "langchain"})}")
#print("=========\n")

# ### Routing
# 
# In lesson 3, we show an example of function calling deciding between two candidate functions.
# 
# Given our tools above, let's format these as OpenAI functions and show this same behavior.


functions = [
    convert_to_openai_function(f) for f in [
        search_wikipedia, get_current_temperature
    ]
]
model = ChatOpenAI(api_key=api_key, temperature=0).bind(functions=functions)


#print(f"SF weather: {model.invoke("what is the weather in sf right now")}")
#print(f"Ask something: {model.invoke("what is langchain")}")


from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant"),
    ("user", "{input}"),
])

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

def ask_question(chain, question: str) -> str:
    print(f"Question: {question}" )
    result = chain.invoke({"input": question})
    print(result)
    print("=========")


from langchain.schema.agent import AgentFinish
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        logger.debug(f"Use tool: {result.tool}")
        tools = {
            "search_wikipedia": search_wikipedia, 
            "get_current_temperature": get_current_temperature,
        }
        return tools[result.tool].run(result.tool_input)

chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route

ask_question(chain, "What is the weather in Boston now?")
ask_question(chain, "Who is the current mayor of SF?")
ask_question(chain, "What is 1000 + 20?")


