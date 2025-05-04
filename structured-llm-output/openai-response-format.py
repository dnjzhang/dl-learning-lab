#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')

from typing import List, Literal, Optional
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI

from utils.load_properties import LoadProperties

# Pydantic model for a user
class User(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


# Pydantic model for social media mentions
class Mention(BaseModel):
    product: Literal['app', 'website', 'not_applicable']
    sentiment: Literal['positive', 'negative', 'neutral']
    needs_response: bool
    response: Optional[str]
    support_ticket_description: Optional[str]


def analyze_mention(
        client: OpenAI,
        mention: str,
        personality: str = "friendly"
) -> Mention:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""
                Extract structured information from 
                social media mentions about our products.

                Provide
                - The product mentioned (website, app, not applicable)
                - The mention sentiment (positive, negative, neutral)
                - Whether to respond (true/false). Don't respond to 
                  inflammatory messages or bait.
                - A customized response to send to the user if we need 
                  to respond.
                - An optional support ticket description to create.

                Your personality is {personality}.
            """},
            {"role": "user", "content": mention},
        ],
        response_format=Mention,
    )
    return completion.choices[0].message.parsed


if __name__ == "__main__":
    # Load API key
    properties = LoadProperties()
    api_key = properties.getApiKey()

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    mentions = [
        "@techcorp your app is amazing!",
        "@techcorp website is down, please fix.",
        "hey @techcorp you're so evil"
    ]

    # Process mentions
    rows: List[dict] = []
    for m in mentions:
        result = analyze_mention(client, m)
        # print_mention(result, m)
        data = result.model_dump()
        data['mention'] = m
        rows.append(data)

    df = pd.DataFrame(rows)
    print(df)
