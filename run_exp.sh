port=8000
N=0
seed=20231201
D=("icews_wiki")

for d in ${D[*]}
do
echo "pre-obtain description of $d"
python preobtain_description.py --random_seed $seed --port $port --data $d --desc_file description --cand_file cand --neigh 25 --ent $N --max_tokens 80
echo "process $d base"
python main_ChatEA.py --random_seed $seed --LLM llama --port $port --data $d --desc_file description --cand_file cand --neigh 5 --ent $N
done