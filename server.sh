TP=4
SP=1

input=500
output=1
NUM=20
QPS=20


NCCL_SOCKET_IFNAME=eth0 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python test/longserve/5-start-api-server.py --backend longserve-fixsp -tp $TP -sp $SP --dataset sharegpt \
    > logs/server-tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS.log &

