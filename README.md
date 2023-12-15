# ChatEA

This is code for ChatEA

### Dependencies

```
openai==0.28.1
```

### How to Run

The model runs in three steps:

#### 1. Collect Candidates

Use EA methods based on emebddings to obtain the candidate entities, which are  the top 20 entities in the alignment results.

save the candidate entities, the name dictionary, and the knowledge tuples under`data/{DATASET}/candidates`.

For the format of data, please see `cand`, `name_dict` and `neighbors` under [data/example/candidates](data/example/candidates)

#### 2. Pre-obtain the entity descrptions

For efficiency, we need to pre-obtain the entity descriptions using LLM based on tuples in knowledge graph and the inherent knowldege from LLM itself.

To  pre-obtain the entity descriptions, use:

```bash
python preobtain_description.py \
	--data icews_wiki \		
	--desc_file description \
	--cand_file cand \
	--neigh 25 \
	--ent 0 \
	--max_tokens 80
```

#### 3. Run ChatEA

To run ChatEA and evaluation code, use:

```bash
python main_ChatEA.py \
	--LLM llama	\
	--port 8000 \
	--data icews_wiki \
	--cand_file cand \
	--desc_file description \
	--neigh 5 \
	--ent 0
```

use `--log_print` to output the prompt and response of LLM

use `--save_step X` to save result for each `X` entities

Or you can use:

```bash
bash run_exp.sh
```

to directly run step 2 and step 3.

