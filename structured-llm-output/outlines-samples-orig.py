#!/usr/bin/env python
# coding: utf-8

# # L5: Structured Generation: Beyond JSON!
# 

# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from utils import template
import outlines
from outlines.samplers import greedy


# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>.
# 
# <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>
# 
# <p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
# </div>

# In[ ]:


model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = outlines.models.transformers(model_name)


# ## Choice

# In[ ]:


prompt = template("""
Look at this restaurant review and classify its sentiment. 
Respond only with 'positive' or 'negative':

Review: The pizza a the was delicious, and the service was excellent.
""")


# In[ ]:


sentiment_regex = r'(positive|negative)'


# In[ ]:


chooser = outlines.generate.choice(
    model,
    ['positive', 'negative'],
    sampler=greedy()
)


# In[ ]:


chooser(prompt)


# ## Phone number

# In[ ]:


phone_prompt = template("""
Extract the phone number from the example,
please use the format: (XXX) XXX-XXXX

206-555-1234

""")


# In[ ]:


phone_regex = r'\([0-9]{3}\) [0-9]{3}-[0-9]{4}'


# In[ ]:


phone_generator = outlines.generate.regex(
    model, 
    phone_regex,
    sampler=greedy()
)


# In[ ]:


phone_generator(phone_prompt)


# ## Email Address

# In[ ]:


email_regex = r'[a-zA-Z0-9]{3,10}@[a-z]{4,20}\.com'


# In[ ]:


email_prompt = template("Give me an email address for someone at amazon")


# In[ ]:


email_generator = outlines.generate.regex(
    model,
    email_regex,
    sampler=greedy())


# In[ ]:


email_generator(email_prompt)


# ## HTML Image Tag

# In[ ]:


example = '<img src="large_dinosaur.png" alt="Image of Large Dinosaur">'


# In[ ]:


img_tag_regex = r'<img src="\w+\.(png|jpg|gif)" alt="[\w ]+">'


# In[ ]:


import re

print(re.search(img_tag_regex, example)[0])


# In[ ]:


img_tag_generator = outlines.generate.regex(model, img_tag_regex)


# In[ ]:


img_tag = img_tag_generator(
    template(
        """Generate a basic html image tag for the file 'big_fish.png', 
        make sure to include an alt tag"""
    ))


# In[ ]:


print(img_tag)


# In[ ]:


from IPython.display import HTML, display

display(HTML(img_tag))


# ## Tic-Tac-Toe

# In[ ]:


ttt_regex = r'[XO ]\|[XO ]\|[XO ]\n-\+-\+-\n[XO ]\|[XO ]\|[XO ]\n-\+-\+-\n[XO ]\|[XO ]\|[XO ]'


# In[ ]:


ttt_generator = outlines.generate.regex(model, ttt_regex, sampler=greedy())


# In[ ]:


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


# In[ ]:


print(ttt_out)


# ## CSV

# In[ ]:


csv_regex = r'Code,Amount,Cost\n([A-Z]{3},[1]*[0-9],1]*[0-9]\.[0-9]{2}\n){1,3}'


# In[ ]:


csv_generator = outlines.generate.regex(model, csv_regex)


# In[ ]:


csv_out = csv_generator(
    template(
        """Create a CSV file for 2-3 store inventory items.
           Include a column 'Code', 'Amount', and 'Cost'.
        """)
)


# In[ ]:


from io import StringIO
import pandas as pd
pd.read_csv(StringIO(csv_out))


# ## GSM8K and Making REGEX easier!

# ```
# Question: Tom has 3 cucumbers, Joes gives him 2 more cucumbers, 
#           how many does Tom have?
# Reasoning: Tom started with 3 cucumbers, then received 2 more. 
#            This means he has 5 cucumbers.
# So the answer is: 5
# ```

# In[ ]:


from outlines.types import sentence, digit
from outlines.types.dsl import to_regex

# Write between 1-3 Sentences
reasoning = "Reasoning: " + sentence.repeat(1,2)
# Answer in 1-4 digits
answer = "So the answer is: " + digit.repeat(1,4)

to_regex(reasoning + "\n" + answer)


# In[ ]:


gsm8k_generator = outlines.generate.regex(
    model, 
    to_regex(reasoning + "\n" + answer),
    sampler=greedy()
)


# In[ ]:


question = """
Sally has 5 apples, then received 2 more, how many apples does Sally have?
"""


# In[ ]:


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


# In[ ]:


result = gsm8k_generator(question_prompt)

print(result)


# # Build Your Own Hotdog vs. Not a hotdog

# In[ ]:


from transformers import AutoProcessor
from utils import load_and_resize_image, get_messages
from transformers import AutoModelForVision2Seq
import torch

vmodel_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
model_class = AutoModelForVision2Seq

vmodel = outlines.models.transformers_vision(
    vmodel_name,
    model_class=model_class,
)

# Used for generating prompt
processor = AutoProcessor.from_pretrained(vmodel_name)


# In[ ]:


hotdog_or_not = outlines.generate.text(
    vmodel,
    sampler=greedy()
)


# In[ ]:


base_prompt="""
You are being given of an image that is either of a
 hotdog
or
 not a hotdog
You must correctly label this. Repond with only "hotdog" or "not a hotdog"
"""


# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Note)</code>:</b>The following cell processes multiple images and may take a while to complete.</p>

# In[ ]:


for i in range(1,6):
    image = load_and_resize_image(f"./hotdog_or_not/{i}.png")
    prompt = processor.apply_chat_template(
        get_messages(image,base_prompt=base_prompt), 
        tokenize=False, 
        add_generation_prompt=True
    )
    print(hotdog_or_not(prompt, [image]))
    display(image)
    print("-------")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




