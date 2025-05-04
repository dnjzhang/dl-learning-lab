#!/usr/bin/env python

# # L2: How To Use Structured Outputs

# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# In[1]:


# Warning control
import warnings
warnings.filterwarnings('ignore')

from utils.load_properties import LoadProperties
properties = LoadProperties()
KEY = properties.getApiKey()

# In[2]:


import os



# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>.
# 
# <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>
# 
# <p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
# </div>

# In[3]:


from openai import OpenAI

# Instantiate the client
client = OpenAI(
    api_key=KEY
)


# ## Define structure with Pydantic

# In[4]:


# The user class from the slides
from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


# In[5]:


completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Make up a user."},
    ],
    response_format=User,
)


# In[6]:

user = completion.choices[0].message.parsed
print( f"User: {user}" )


# ## The social media mention structure

# In[7]:


from pydantic import BaseModel
from enum import Enum
from typing import List, Optional, Literal
from openai import OpenAI

class Mention(BaseModel):
    # The model chooses the product the mention is about,
    # as well as the social media post's sentiment
    product: Literal['app', 'website', 'not_applicable']
    sentiment: Literal['positive', 'negative', 'neutral']

    # Model can choose to respond to the user
    needs_response: bool
    response: Optional[str]

    # If a support ticket needs to be opened, 
    # the model can write a description for the
    # developers
    support_ticket_description: Optional[str]


# In[8]:


# Example mentions
mentions = [
    # About the app
    "@techcorp your app is amazing! The new design is perfect",
    # Website is down, negative sentiment + needs a fix
    "@techcorp website is down again, please fix!",
    # Nothing to respond to
    "hey @techcorp you're so evil"
]


# In[9]:


def analyze_mention(
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


# In[10]:


print("User post:", mentions[0])
processed_mention = analyze_mention(mentions[0])
processed_mention


# In[11]:


rude_mention = analyze_mention(mentions[0], personality="rude")
rude_mention.response


# In[12]:


mention_json_string = processed_mention.model_dump_json(indent=2)
print(mention_json_string)


# ## You try!

# In[13]:


class UserPost(BaseModel):
    message: str

def make_post(output_class):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """
                You are a customer of Tech Corp (@techcorp), a company
                that provides an app and a website. Create a small 
                microblog-style post to them that sends some kind of 
                feedback, positive or negative.
            """},
            {"role": "user", "content": "Please write a post."},
        ],
        response_format=output_class,
    )
    return completion.choices[0].message.parsed

new_post = make_post(UserPost)
new_post


# In[14]:


analyze_mention(new_post.message)


# In[15]:


class UserPostWithExtras(BaseModel):
    user_mood: Literal["awful", "bad", "evil"]
    product: Literal['app', 'website', 'not_applicable']
    sentiment: Literal['positive', 'negative', 'neutral']
    internal_monologue: List[str]
    message: str
    
new_post = make_post(UserPostWithExtras)
new_post


# In[16]:


analyze_mention(new_post.message)


# ## Programming with our mentions

# In[17]:


from helper import print_mention

# Loop through posts that tagged us and store the results in a list
rows = []
for mention in mentions:
    # Call the LLM to get a Mention object we can program with
    processed_mention = analyze_mention(mention)

    # Print out some information
    print_mention(processed_mention, mention)
    
    # Convert our processed data to a dictionary
    # using Pydantic tools
    processed_dict = processed_mention.model_dump()
    
    # Store the original message in the dataframe row
    processed_dict['mention'] = mention
    rows.append(processed_dict)
    
    print("") # Add separator to make it easier to read


# In[18]:


import pandas as pd

df = pd.DataFrame(rows)
df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




