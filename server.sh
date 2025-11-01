TP=1
SP=1

input=500
output=1
NUM=20
QPS=20

export LWM_WEIGHT_PATH=/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export LWM_WEIGHT_DISTSERVE_PATH=/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export EXP_RESULT_ROOT_PATH=/workspace/result

NCCL_SOCKET_IFNAME=eth0 \
CUDA_VISIBLE_DEVICES=7 \
    python test/longserve/5-start-api-server.py --backend longserve-fixsp -tp $TP -sp $SP --dataset sharegpt \
    > logs/server-tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS.log &

