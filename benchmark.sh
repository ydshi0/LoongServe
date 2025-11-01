TP=2
SP=2

input=500
output=1
NUM=20
QPS=25

Backend=longserve-fixsp

export LWM_WEIGHT_PATH=/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export LWM_WEIGHT_DISTSERVE_PATH=/shared_LLM_model/meta-llama/Meta-Llama-3.1-8B-Instruct
export EXP_RESULT_ROOT_PATH=/workspace/result

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9   
  pgrep python3 | xargs -r kill -9          
  pkill -f zmq
  for port in 8700 10003 10004; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

kill_gpu_processes

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python test/longserve/5-start-api-server.py --backend $Backend -tp $TP -sp $SP --dataset sharegpt \
> logs/server-tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS.log &


sleep 60

python test/longserve/5-benchmark-serving.py --backend $Backend --dataset /workspace/ydshi/dataset/sharegpt.ds --num-prompts-req-rates "[($NUM, $QPS)]" --inputlen $input --outputlen $output > logs/tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS.log

kill_gpu_processes



