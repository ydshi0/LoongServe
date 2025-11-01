#!/bin/bash
set -ex
TP=2
SP=2
WORLD_SIZE=$((TP * SP))
input=2000
output=2000
NUM=20
QPS=0.5

BACKEND=longserve-fixsp
PORT=18700
MODEL=meta-llama/Llama-2-7b-chat-hf
# MODEL=meta-llama/Llama-2-13b
MODEL_PATH=/shared_LLM_model/$MODEL
MODEL_BASENAME=$(basename "$MODEL")
LOG_PREFIX=logs/${MODEL_BASENAME}/tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS
mkdir -p $LOG_PREFIX


unset NCCL_SOCKET_IFNAME
kill_gpu_processes() {
  # kill all processes on GPU.
  # pgrep pt_main_thread | xargs -r kill -9   
  pgrep python3 | xargs -r kill -9          
  for port in 18700; do lsof -t -i:$port | xargs -r kill -9; done
  ray stop
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/health > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


kill_gpu_processes
sleep 3

CUDA_VISIBLE_DEVICES=4,5,6,7 \
ray start --head \
  --port=24031 \
  --dashboard-port=24032 \
  --redis-shard-ports=24033,24034 \
  --node-manager-port=24035 \
  --object-manager-port=24036 \
  --disable-usage-stats \
  --temp-dir=/tmp/ray_head_1

# python test/longserve/5-start-api-server.py --backend $BACKEND -tp $TP -sp $SP --dataset sharegpt \
#   > logs/server-tp${TP}-sp${SP}-$input-$output-num$NUM-$QPS.log &
python -u -m loongserve.longserve_server.api_server \
    --host 0.0.0.0 --port $PORT \
    --model_dir $MODEL_PATH \
    --tokenizer_mode auto \
    --max_total_token_num 80000 --running_max_req_size 30 \
    --tp_world_size $TP --sp_world_size $SP \
    --max_req_input_len 41999 --max_req_total_len 42000 \
    --mode _token_decode_attention_overlapped \
    --batch_max_tokens 160000 \
    --max_mig_len 10000 \
    --avg_decoding_time 22 \
    --nccl_port 28768 \
    --log_stats_interval 600 \
    --max_prefill_time 500000 \
    --local_world_size $WORLD_SIZE \
    --max_wait_tokens 10 \
    --min_comp_bound_decoding_batch_size 128 \
    --profiler_file_path /workspace/result/analytical-model.csv \
    --max_num_ooe 1 --use_fixed_sp 2>&1 > $LOG_PREFIX/server.log &

wait_for_server 18700
sleep 3


export LWM_WEIGHT_PATH=$MODEL_PATH
export LWM_WEIGHT_DISTSERVE_PATH=$MODEL_PATH
export EXP_RESULT_ROOT_PATH=/serve/LoongServe/result
python test/longserve/5-benchmark-serving.py --port $PORT --backend $BACKEND --dataset /workspace/ydshi/dataset/sharegpt.ds --num-prompts-req-rates "[($NUM, $QPS)]" --inputlen $input --outputlen $output 2>&1 > $LOG_PREFIX/client.log 

# kill_gpu_processes



