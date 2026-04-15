
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
SHORT_NAME=mistral-7B
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
SHORT_NAME=llama-3.1-8B
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
SHORT_NAME=llama-3.2-3B
#MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
#SHORT_NAME=llama-3.3-70B
#ID_EXPERIMENT=580024
ID_EXPERIMENT=public
AGENT_TYPE=aggregator 
#PERSONA=private_user #--persona ${PERSONA} 
#VARIATION=replace #--prompt_variation ${VARIATION}
MODEL_ANSWER_POST=gpt-4o
#for INIT in 0 1 2 3 4 5 6 7 8

#for INIT in 0 1 2 3 4 5 6 7
for INIT in 0

# --num_datapoints 4600
do
    DEVICE=1
    screen -dmS ${AGENT_TYPE}_${SHORT_NAME}_${INIT}_${ID_EXPERIMENT} python experiments/call_llm.py --num_datapoints 4600 --device ${DEVICE} --agent_type ${AGENT_TYPE} --model_answer_post ${MODEL_ANSWER_POST} --id_experiment ${ID_EXPERIMENT} --model_name ${MODEL_NAME} --n_init ${INIT} --num_splits 1 #27
done
#bash scripts/aggregator.sh
