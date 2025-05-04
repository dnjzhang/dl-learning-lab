#!/usr/bin/env python
# coding: utf-8
# # LangChain Expression Language (LCEL)

from utils.load_properties import LoadProperties

properties = LoadProperties()
api_key = properties.getApiKey()

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

# ## Simple Chain
print("Tell Joke:")
prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)
model = ChatOpenAI(api_key=api_key, )
output_parser = StrOutputParser()
chain = prompt | model | output_parser

print("Joke: " + chain.invoke({"topic": "bears"}))

# ## More complex chain
# 
print("And Runnable Map to supply user-provided inputs to the prompt.")
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

def print_doc(docs):
    for i, doc in enumerate(docs, start=1):
        print(f"Result {i}")
        print("-" * (8 + len(str(i))))
        print(doc.page_content)
        if doc.metadata:
            print("\nMetadata:", doc.metadata)
        print("\n")

# mock vector store
vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings(api_key=api_key),
)
retriever = vectorstore.as_retriever()

print_doc(retriever.get_relevant_documents("where did harrison work?"))

print_doc(retriever.get_relevant_documents("what do bears like to eat"))

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnableMap

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

chain.invoke({"question": "where did harrison work?"})

inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

inputs.invoke({"question": "where did harrison work?"})

# ## Bind
# 
# and OpenAI Functions

functions = [
    {
        "name": "weather_search",
        "description": "Search for weather given an airport code",
        "parameters": {
            "type": "object",
            "properties": {
                "airport_code": {
                    "type": "string",
                    "description": "The airport code to get the weather for"
                },
            },
            "required": ["airport_code"]
        }
    }
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
)
model = ChatOpenAI(api_key=api_key, temperature=0).bind(functions=functions)

runnable = prompt | model

runnable.invoke({"input": "what is the weather in sf"})

functions = [
    {
        "name": "weather_search",
        "description": "Search for weather given an airport code",
        "parameters": {
            "type": "object",
            "properties": {
                "airport_code": {
                    "type": "string",
                    "description": "The airport code to get the weather for"
                },
            },
            "required": ["airport_code"]
        }
    },
    {
        "name": "sports_search",
        "description": "Search for news of recent sport events",
        "parameters": {
            "type": "object",
            "properties": {
                "team_name": {
                    "type": "string",
                    "description": "The sports team to search for"
                },
            },
            "required": ["team_name"]
        }
    }
]

model = model.bind(functions=functions)

runnable = prompt | model

runnable.invoke({"input": "how did the patriots do yesterday?"})

# ## Fallbacks


from langchain_openai import OpenAI
import json

# **Note**: Due to the deprication of OpenAI's model `text-davinci-001` on 4 January 2024, you'll be using OpenAI's recommended replacement model `gpt-3.5-turbo-instruct` instead.


simple_model = ChatOpenAI(
    api_key=api_key,
    temperature=0,
    max_tokens=1000,
    model="gpt-4o-mini"
)
simple_chain = simple_model | json.loads

challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"

simple_model.invoke(challenge)

# <p style=\"background-color:#F5C780; padding:15px\"><b>Note:</b> The next line is expected to fail.</p>

#simple_chain.invoke(challenge)

model = ChatOpenAI(api_key=api_key, temperature=0)
chain = model | StrOutputParser() | json.loads

chain.invoke(challenge)

final_chain = simple_chain.with_fallbacks([chain])

final_chain.invoke(challenge)

# ## Interface


prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
#model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "bears"})

chain.batch([{"topic": "bears"}, {"topic": "frogs"}])

for t in chain.stream({"topic": "bears"}):
    print(t)

import asyncio


async def main():
    response = await chain.ainvoke({"topic": "bears"})
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
