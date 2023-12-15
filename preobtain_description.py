import os
import random
import json
import time
import argparse
import openai
from tqdm import tqdm
from tools_for_ChatEA import *


engine = ""
no_random = False


def try_get_response(prompt, messages=[], max_tokens=100, max_try_num=3):
	try_num = 0
	flag = True
	response = None
	while flag:
		try:
			# request LLM
			response = openai.ChatCompletion.create(
				model=engine,
				messages=messages + [{"role": "user", "content": prompt}],
				max_tokens=max_tokens,
				temperature=0.2
			)
			flag = False
		except openai.OpenAIError as e:
			try_num += 1
			if try_num >= max_try_num:
				break
	return response, (not flag)


def generate_prompt(entity):
    neighbors = [f"({', '.join(list(neigh))})" for neigh in entity["neighbors"]]

    prompt = f"Your task is to give a one-sentence brief introduction for given [Entity], based on 1.YOUR OWN KNOWLEDGE; 2.[Knowledge Tuples]. NOTICE, introduction is just one sentence and less than 50 tokens."
    prompt += "Here is a example:\n[KNOWLEDGE]: Given [Entity] Gun Hellsivik and its related [Knowledge Tuples]: [(Gun Hellsvik, member of, Moderate Party)].\n[Input] What is Gun Hellsivik? Please give a one-sentence brief introduction based on YOUR OWN KNOWLEDGE and [Knowledge Tuples]\n[Output]: Gun Hellsvik was a Swedish politician and member of the Moderate Party, knownfor serving as the Minister of Justice in Sweden.\n"
    prompt += "Now please answer:\n"
    prompt += f"[KNOWLEDGE]: Given [Entity] {entity['name']} and its related [Knowledge Tuples]: [{', '.join(neighbors)}].\n"
    prompt += f"[Input]: What is {entity['name']}? Please give a one-sentence brief introduction based on YOUR OWN KNOWLEDGE and [Knowledge Tuples].\n"
    prompt += "[Output]: "

    return prompt


def process_res(res:str):
    ###############
    ### extract entity description from repsonse
    ###############
    text_list = res.strip().split(":")
    text = ":".join(text_list[1:]) if len(text_list) > 1 else text_list[0]
    text = text.strip().split("\n")[0]
    return text.strip()

def get_description(idx, ng:NeighborGenerator, entities, neigh=0, max_tokens=50):
    description = {}
    for eid in tqdm(entities, desc=f"{idx:2d}", position=idx):
        ent = ng.get_neighbors(eid, neigh)
        description[eid] = {"name": ent["name"], "desc": ""}
        res, get_res = try_get_response(generate_prompt(ent), max_tokens=max_tokens)
        desc_text = res["choices"][0]["message"]["content"] if get_res else "[ERROR]"
        description[eid]["desc"] = process_res(desc_text)
    return description


def generate_entity_description(data, data_dir, cand_file="cand", desc_file="description", ent=0, neigh=0, max_tokens=50, use_time=True):
    ng = NeighborGenerator(data=data, data_dir=data_dir, use_time=use_time, use_desc=False, cand_file=cand_file, desc_file=desc_file)
    entities = ng.get_all_entities()
    if ent > 0:
        if not no_random:
            random.shuffle(entities)
        entities = entities[:ent]
    st = time.time()
    
    description = get_description(0, ng, entities, neigh, max_tokens)

    time_cost = time.time() - st
    h, m, s = transform_time(int(time_cost))
    print(f'Time Cost : {h}hour, {m:02d}min, {s:02d}sec')

    return description


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--data", type=str, default="icews_wiki")
parser.add_argument("--desc_file", type=str, default="description")
parser.add_argument("--cand_file", type=str, default="cand")
parser.add_argument("--neigh", type=int, default=25)
parser.add_argument("--ent", type=int, default=0)
parser.add_argument("--max_tokens", type=int, default=80)
parser.add_argument("--no_random", action="store_true")
parser.add_argument("--random_seed", type=int, default=20231201)
args = parser.parse_args()

### random setting
no_random = args.no_random
os.environ['PYTHONHASHSEED'] = str(args.random_seed)
random.seed(args.random_seed)

### change these settings, according to your LLM 
openai.api_base = f"http://localhost:8000/v1"
openai.api_key = "EMPTY"
engine = openai.Model.list()["data"][0]["id"]

### generate entity description previously
use_time = True if args.data in ["icews_wiki", "icews_yago"] else False

description = generate_entity_description(data=args.data, data_dir=args.data_dir, cand_file=args.cand_file, desc_file=args.desc_file, ent=args.ent, neigh=args.neigh, max_tokens=args.max_tokens, use_time=use_time)

### save entity description
with open(os.path.join(args.data_dir, args.data, "candidates", args.desc_file), "w", encoding="utf-8") as fw:
    json.dump(description, fw, ensure_ascii=False, indent=4)

