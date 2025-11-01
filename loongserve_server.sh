TP=2
SP=4

input=500
output=1
NUM=20
QPS=20
export NCCL_DEBUG=DEBUG
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH
export LWM_WEIGHT_PATH=/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export LWM_WEIGHT_DISTSERVE_PATH=/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export EXP_RESULT_ROOT_PATH=/workspace/result

# ray start --head 

NCCL_SOCKET_IFNAME=eth0 \
CUDA_VISIBLE_DEVICES=7\
    python test/longserve/5-start-api-server.py --backend longserve -tp $TP -sp $SP --dataset sharegpt \
    > logs/server-tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS.log `