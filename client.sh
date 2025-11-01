TP=1
SP=1

input=500
output=1
NUM=10
QPS=20

export LWM_WEIGHT_PATH=/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export LWM_WEIGHT_DISTSERVE_PATH=/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export EXP_RESULT_ROOT_PATH=/workspace/result

python test/longserve/5-benchmark-serving.py --backend longserve-fixsp --dataset /workspace/ydshi/dataset/sharegpt.ds --num-prompts-req-rates "[($NUM, $QPS)]" --inputlen $input --outputlen $output > logs/tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS.log
