TP=1
SP=2

input=500
output=1
NUM=100
QPS=20


export LWM_WEIGHT_PATH=/host_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export LWM_WEIGHT_DISTSERVE_PATH=/host_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export EXP_RESULT_ROOT_PATH=/host_home/Loongserve/exp_result
python test/longserve/5-benchmark-serving.py --backend longserve --dataset /host_home/dataset/vllm/sharegpt/sharegpt.ds --num-prompts-req-rates "[($NUM, $QPS)]" --inputlen $input --outputlen $output > logs/tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS.log
