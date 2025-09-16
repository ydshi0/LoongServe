TP=4
SP=1

input=500
output=1
NUM=10
QPS=20



python test/longserve/5-benchmark-serving.py --backend longserve-fixsp --dataset /workspace/ydshi/dataset/sharegpt.ds --num-prompts-req-rates "[($NUM, $QPS)]" --inputlen $input --outputlen $output > logs/tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS.log
