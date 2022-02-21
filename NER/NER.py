"""
This script runs inference of T0++ on multiple GPUs using model parallelism.
# The minimum requirements to run T0++ (11B parameters) inference are 4 16GB V100 or 2 32GB V100 (or basically, enough GPU memory to fit ~42GB of fp32 parameters).
# (https://github.com/bigscience-workshop/t-zero/blob/master/inference/model_parallelism.py)

"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
import torch
import pandas
import re

# List of entity types
ent_types = ["PERS", "LOC", "ORG", "TIME", "PROD.MED", "PROD.DOC"] # I split products into media and doctrines as "products" is too general as a prompt

# Upload model
model_name = "bigscience/T0pp"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded")

model.parallelize()
print("Moved model to GPUs")

# Inference function
def T0_infer(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(inputs)

    answer = (tokenizer.decode(outputs[0], skip_special_tokens=True))

"TO DO: upload datset"

"TO DO: cycle on sentences"
# Import sentence
# Clean sentence: remove at least split words (like "in", "-", "dustrious")
# Regularise spelling
sentence = sentence

# Cycle over prompts
df = pd.DataFrame(columns=["token", "entity_type"])
ent_types_query = ["names of person", "names of location", "names of organization", "dates", "names of media", "names of doctrine"]
for k in ent_types:
    prompt = "Input: {}.format(sentence)" + "\n" + "In input, what are the {}? Separate answers with commas".format(ent_types_query)
    answer = T0_infer(prompt)
    inferred_ents = answer.split(",")

    # Check if token exists and add it to dataframe    
    for token in inferred_ents:
        if re.search(token, sentence, re.IGNORECASE):
            df = df.append({"token": token, "entity_type": ent_types[k]}, ignore_index=True)

# Disambiguate 1: choose longest token when token is substring of another token
lower_span_idx = []
for i in range(len(df)):
    for j in range(len(df)):
        if i != j:
            if re.search(df["token"][j], df["token"][i], re.IGNORECASE):
                lower_span_idx.append(j)
df = df.drop(lower_span_idx)

# Disambiguate 2: disambiguate duplicate tokens
df_dedup = df.groupby('token')['entity_type'].apply(lambda x: list(x)).reset_index()
for i in range(len(df_dedup)):
    if len(df_duplicates["entity"][i]) > 1:
        token_dup = df_dedup["token"][i]
        ent_dup = df_duplicates["entity"][i]

        for i,n in enumerate(ent_dup): # replace NER codes with text for prompt
            if n == 'PERS':
                ent_dup[i] = 'person'
            elif n == 'LOC':
                ent_dup[i] = 'location'
            elif n == 'ORG':
                ent_dup[i] = 'organization'
            elif n == 'TIME':
                ent_dup[i] = 'date'
            elif n == 'PROD.MED':
                ent_dup[i] = 'media'
            elif n == 'PROD.DOC':
                ent_dup[i] = 'doctrine'

        if len(ent_duplicates) == 2:
            prompt = "Input: {}.format(sentence)" + "\n" + "In input, is \"{}\", a {} or a {}? Give only one answer".format(token_dup, ent_dup[0], ent_dup[1])
        elif len(ent_duplicates) == 3:
            prompt = "Input: {}.format(sentence)" + "\n" + "In input, is \"{}\", a {}, a {} or a {}? Give only one answer".format(token_dup, ent_dup[0], ent_dup[1], ent_dup[2])
        elif len(ent_duplicates) == 4:
            prompt = "Input: {}.format(sentence)" + "\n" + "In input, is \"{}\", a {}, a {}, a {} or a {}?. Give only one answer".format(token_dup, ent_dup[0], ent_dup[1], ent_dup[2], ent_dup[3])
        elif len(ent_duplicates) == 5:
            prompt = "Input: {}.format(sentence)" + "\n" + "In input, is \"{}\", a {}, a {}, a {}, a {} or a {}? Give only one answer".format(token_dup, ent_dup[0], ent_dup[1], ent_dup[2], ent_dup[3], ent_dup[4])
        elif len(ent_duplicates) == 6:
            prompt = "Input: {}.format(sentence)" + "\n" + "In input, is \"{}\", a {}, a {}, a {}, a {}, a {} or a {}? Give only one answer".format(token_dup, ent_dup[0], ent_dup[1], ent_dup[2], ent_dup[3], ent_dup[4], ent_dup[5])

    answer = T0_infer(sentence)

    # Check that answer is actually an entity type
    if answer in ent_types:
        df_duplicates["entity"][i] = [answer]
    else:
        df_dedup.drop(i, inplace=True)

df_dedup["entity"] = df_dedup["entity"].apply(lambda x: x[0])
df = df_dedup

"TO DO: collapse PROD.MED and PROD.DOC into PROD"

"TO DO: add predicted entity to dataframe with all predictions"
