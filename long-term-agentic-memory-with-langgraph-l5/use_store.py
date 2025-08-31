
from langgraph.store.memory import InMemoryStore
#!/usr/bin/env python
# coding: utf-8
import argparse

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
_ = load_dotenv()


store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)
