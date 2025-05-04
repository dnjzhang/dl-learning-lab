#!/usr/bin/env python
# coding: utf-8

# # L5: Structured Generation: Beyond JSON!
#

import warnings

warnings.filterwarnings('ignore')

from utils_local import template
import outlines
from outlines.samplers import greedy

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = outlines.models.transformers(model_name)

review = "The pizza a the was delicious, and the service was excellent."
prompt = template("""
Look at this restaurant review and classify its sentiment. 
Respond only with 'positive' or 'negative':

Review: """ + review)

sentiment_regex = r'(positive|negative)'

chooser = outlines.generate.choice(
    model,
    ['positive', 'negative'],
    sampler=greedy()
)

print(f"Sentiment for \"{review}\": {chooser(prompt)}")
print("===")

phone_prompt = template("""
Extract the phone number from the example,
please use the format: (XXX) XXX-XXXX

206-555-1234

""")

phone_regex = r'\([0-9]{3}\) [0-9]{3}-[0-9]{4}'
phone_generator = outlines.generate.regex(
    model,
    phone_regex,
    sampler=greedy()
)

print("Extracted phone number: " + phone_generator(phone_prompt))
print("===")

email_regex = r'[a-zA-Z0-9]{3,10}@[a-z]{4,20}\.com'
email_prompt = template("Give me an email address for someone at amazon")

email_generator = outlines.generate.regex(
    model,
    email_regex,
    sampler=greedy())

print("Generated Amazon phone number: " + email_generator(email_prompt))
print("===")

example = '<img src="large_dinosaur.png" alt="Image of Large Dinosaur">'
img_tag_regex = r'<img src="\w+\.(png|jpg|gif)" alt="[\w ]+">'

import re

print(re.search(img_tag_regex, example)[0])

img_tag_generator = outlines.generate.regex(model, img_tag_regex)

img_tag = img_tag_generator(
    template(
        """Generate a basic html image tag for the file 'big_fish.png', 
        make sure to include an alt tag"""
    ))

print(f"Generate image tag for big_fish.png: {img_tag}")
from bs4 import BeautifulSoup
from PIL import Image

html = img_tag
# 1) parse out the src
soup = BeautifulSoup(html, "html.parser")
img_src = soup.img["src"]
img = Image.open(img_src)
img.show()  # launches your OS’s default image viewer
print("===")

# ## Tic-Tac-Toe

ttt_regex = r'[XO ]\|[XO ]\|[XO ]\n-\+-\+-\n[XO ]\|[XO ]\|[XO ]\n-\+-\+-\n[XO ]\|[XO ]\|[XO ]'

ttt_generator = outlines.generate.regex(model, ttt_regex, sampler=greedy())

ttt_out = ttt_generator("""
We'll be representing an ASCII tic-tac-toe board like this:
```
 | | 
-+-+-
 | | 
-+-+-
 | | 
```
With X,O or a blank space being valid entries.
Here is an example game that is currently in progress:
"""
                        )

print(ttt_out)
print("===")

# ## CSV

csv_regex = r'Code,Amount,Cost\n([A-Z]{3},[1]*[0-9],1]*[0-9]\.[0-9]{2}\n){1,3}'

csv_generator = outlines.generate.regex(model, csv_regex)

csv_out = csv_generator(
    template(
        """Create a CSV file for 2-3 store inventory items.
           Include a column 'Code', 'Amount', and 'Cost'.
        """)
)

from io import StringIO
import pandas as pd

print("csv")
print(pd.read_csv(StringIO(csv_out)))
print("===")

# ## GSM8K and Making REGEX easier!

# ```
# Question: Tom has 3 cucumbers, Joes gives him 2 more cucumbers, 
#           how many does Tom have?
# Reasoning: Tom started with 3 cucumbers, then received 2 more. 
#            This means he has 5 cucumbers.
# So the answer is: 5
# ```


from outlines.types import sentence, digit
from outlines.types.dsl import to_regex

# Write between 1-3 Sentences

reasoning = "Reasoning: " + sentence.between(1, 2)
# Answer in 1-4 digits
answer = "So the answer is: " + digit.between(1, 4)

to_regex(reasoning + "\n" + answer)

gsm8k_generator = outlines.generate.regex(
    model,
    to_regex(reasoning + "\n" + answer),
    sampler=greedy()
)

question = """
Sally has 5 apples, then received 2 more, how many apples does Sally have?
"""

question_prompt = template(f"""
Please answer the question and the end using the following format:
Example:
Question: Tom has 3 cucumbers, Joes gives him 2 more cucumbers, 
          how many does Tom have?
Reasoning: Tom started with 3 cucumbers, then received 2 more. 
           This means he has 5 cucumbers.
So the answer is: 5

Here is the question you need to answer:
Question: {question}
""")

result = gsm8k_generator(question_prompt)

print(result)
