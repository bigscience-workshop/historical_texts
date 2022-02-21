"""
This script runs inference of T0++ on multiple GPUs using model parallelism.
# The minimum requirements to run T0++ (11B parameters) inference are 4 16GB V100 or 2 32GB V100 (or basically, enough GPU memory to fit ~42GB of fp32 parameters).
# (https://github.com/bigscience-workshop/t-zero/blob/master/inference/model_parallelism.py)

"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas
import re

"TO DO: create function(prompt) for inference and call when needed rather than repeating code"

# List of entity types
ent_types = ["names of person", "names of location", "names of organization", "dates", "names of media", "names of doctrine"] # I split products into media and doctrines

# Upload model
model_name = "bigscience/T0pp"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded")

model.parallelize()
print("Moved model to GPUs")

"TO DO: cycle on languages"

"TO DO: cycle on sentences"
# Import sentence
# Clean sentence
# Regualrise spelling
sentence = sentence

"TO DO: cycle on prompts"
df = pd.DataFrame(columns=["token", "entity_type"])

for entity in ent_types:
    prompt = "Input: {}.format(sentence)" + "\n" + "In input, what are the {}? Separate answers with commas".format(ent_types)

    inputs = tokenizer.encode(prompt_YN, return_tensors="pt")
    inputs = inputs.to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(inputs)

    answer = (tokenizer.decode(outputs[0], skip_special_tokens=True))

    inferred_ents = answer.split(",")

    # Check if token exists and add it to dataframe    
    for token in inferred_ents:
        if re.search(token, sentence, re.IGNORECASE):
            df = df.append({"token": token, "entity_type": entity}, ignore_index=True)

# Disambiguate 1: choose longest toekn when token is substring of another token
lower_span_idx = []
for i in range(len(df)):
    for j in range(len(df)):
        if i != j:
            if re.search(df["token"][j], df["token"][i], re.IGNORECASE):
                lower_span_idx.append(j)
df = df.drop(lower_span_idx)

# Disambiguate 2: disambiguate duplicate tokens
"TO DO: cycle on duplicate tokens"

duplicate = token
ent_duplicates = []

"TO DO: prompts has to be adapted to the number of options (2, 3, 4, 5 or 6)"
    
if len(ent_duplicates) == 2:
    prompt = "Input: {}.format(sentence)" + "\n" + "In input, is \"{}\", a {} or a {}. Give only one answer".format(duplicate, ent_1, ent_2)
elif len(ent_duplicates) == 3:
    prompt = "Input: {}.format(sentence)" + "\n" + "In input, is \"{}\", a {}, a {} or a {}. Give only one answer".format(duplicate, ent_1, ent_2, ent_3)
elif len(ent_duplicates) == 4:
    prompt = "Input: {}.format(sentence)" + "\n" + "In input, is \"{}\", a {}, a {}, a {} or a {}. Give only one answer".format(duplicate, ent_1, ent_2, ent_3, ent_4)
elif len(ent_duplicates) == 5:
    prompt = "Input: {}.format(sentence)" + "\n" + "In input, is \"{}\", a {}, a {}, a {}, a {} or a {}. Give only one answer".format(duplicate, ent_1, ent_2, ent_3, ent_4, ent_5)
elif len(ent_duplicates) == 6:
    prompt = "Input: {}.format(sentence)" + "\n" + "In input, is \"{}\", a {}, a {}, a {}, a {}, a {} or a {}. Give only one answer".format(duplicate, ent_1, ent_2, ent_3, ent_4, ent_5, ent_6)



    inputs = tokenizer.encode(prompt_YN, return_tensors="pt")
    inputs = inputs.to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(inputs)

    answer = (tokenizer.decode(outputs[0], skip_special_tokens=True))
