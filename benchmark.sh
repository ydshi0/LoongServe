TP=1
SP=1

input=500
output=1
NUM=20
QPS=25

Backend=longserve-fixsp

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9   
  pgrep python3 | xargs -r kill -9          
  pkill -f zmq
  for port in 8700 10003; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

kill_gpu_processes


python test/longserve/5-start-api-server.py --backend $Backend -tp $TP -sp $SP --dataset sharegpt \
> logs/server-tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS.log &


sleep 60

python test/longserve/5-benchmark-serving.py --backend $Backend --dataset /workspace/ydshi/dataset/sharegpt.ds --num-prompts-req-rates "[($NUM, $QPS)]" --inputlen $input --outputlen $output > logs/tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS.log

kill_gpu_processes



