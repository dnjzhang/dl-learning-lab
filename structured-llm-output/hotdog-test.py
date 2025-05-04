#!/usr/bin/env python
#Build Your Own Hotdog vs. Not a hotdog
import outlines
from outlines.samplers import greedy

from transformers import AutoProcessor
from utils_local import load_and_resize_image, get_messages
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

hotdog_or_not=r"(hotdog|not hotdog)"
hotdog_or_not = outlines.generate.regex(
    vmodel,
    hotdog_or_not,
    sampler=greedy()
)


base_prompt="""
You are being given of an image that is either of a
 hotdog
or
 not a hotdog
You must correctly label this. Repond with only "hotdog" or "not a hotdog"
"""


for i in range(1,6):
    image = load_and_resize_image(f"./hotdog_or_not/{i}.png")
    prompt = processor.apply_chat_template(
        get_messages(image,base_prompt=base_prompt),
        tokenize=False,
        add_generation_prompt=True
    )
    print(hotdog_or_not(prompt, [image]))
    image.show()
    print("-------")









