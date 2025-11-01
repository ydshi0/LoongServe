TP=2
SP=4

input=500
output=1
NUM=1600
QPS=4
export NCCL_DEBUG=DEBUG
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH
export LWM_WEIGHT_PATH=/host_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export LWM_WEIGHT_DISTSERVE_PATH=/host_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export EXP_RESULT_ROOT_PATH=/host_home/LoongServe/exp_result
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# ray start --head &&\
rm -rf logs/server-tp${TP}-sp${SP}-${input}-${output}-num${NUM}-${QPS}.log
NCCL_SOCKET_IFNAME=eth0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python test/longserve/5-start-api-server.py --backend longserve -tp "${TP}" -sp "${SP}" --dataset sharegpt \
  | tee -a "logs/server-tp${TP}-sp${SP}-${input}-${output}-num${NUM}-${QPS}.log"

# CUDA_VISIBLE_DEVICES=6,7 \
# python3 test/longserve/1-benchmark-identical-req.py longserve-ae-analytical-model-single-node 
# python3 test/longserve/2-find-time-function-prefill.py --profile-db /host_home/LoongServe/exp_result/loongserve-db-identical-req.sqlite --output-csv /host_home/LoongServe/exp_result/analytical-model.csv