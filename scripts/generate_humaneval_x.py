import argparse
from typing import List
import tempfile
import os
import subprocess
from datetime import datetime
import random
from pathlib import Path
from codegeex.benchmark.generate_humaneval_x import initialize
#GLOBALS
  
LANGUAGE: str  
OUTPUT_PATH: str
HOSTLIST: str 

SCRIPT_PATH: str = Path(os.path.abspath(__file__))
print("SCRIPT_PATH = " + str(SCRIPT_PATH))
SCRIPT_DIR: str = os.path.dirname(SCRIPT_PATH)
print("SCRIPT_DIR = " + SCRIPT_DIR)
MAIN_DIR: str = os.path.dirname(SCRIPT_DIR)
print("MAIN_DIR = " + MAIN_DIR)
TOKENIZER_PATH:str = os.path.join(MAIN_DIR, "codegeex/tokenizer/")
print("TOKENIZER_PATH = " + TOKENIZER_PATH)

parser = argparse.ArgumentParser("Debugging generate humaneval_x")

# Target programming language, currently support one of ["python", "java", "cpp", "js", "go"]
parser.add_argument("-l","--language", default="python", type=str)

# Output path of the generated programs.
default_output_path = os.path.join(MAIN_DIR, "codegeex/benchmark/output/humaneval-x/codegeex/")
parser.add_argument("-o","--output_path", default=default_output_path, type=str)


def default_hostlist() -> str:
    ZMQ_ADDR = subprocess.run(["hostname", "-i"], capture_output=True).stdout.decode("utf-8").strip("\n")

    with tempfile.NamedTemporaryFile(dir = SCRIPT_DIR, delete=False) as f:
        os.rename(f.name, os.path.join(SCRIPT_DIR,"hostfile"))
        f.write(ZMQ_ADDR.encode('utf-8'))

    HOSTLIST = os.path.join(SCRIPT_DIR,"hostfile")
    return HOSTLIST

# set master ip for zmq server
# Provide hostfile if generating distributedly
parser.add_argument("-hl","--hostlist", default=default_hostlist(), type=str)

args = parser.parse_args()

LANGUAGE = args.language  
OUTPUT_PATH = args.output_path  
HOSTLIST = args.hostlist
ZMQ_ADDR:str 

with open(HOSTLIST) as f:
    ZMQ_ADDR = f.readline().strip("\n")

# export CUDA settings
os.environ["CUDA_HOME"]= "/usr/local/cuda-11.1/"

#import model configuration
os.environ["CHECKPOINT_PATH"] = "/root/CodeGeeX/codegeex_13b.pt"

os.environ["MODEL_ARGS"] ="""--num-layers 39 \
            --hidden-size 5120 \
            --num-attention-heads 40 \
            --max-position-embeddings 2048 \
            --attention-softmax-in-fp32 \
            --load """ + os.environ["CHECKPOINT_PATH"] + """\
            --layernorm-epsilon 1e-5 \
            --fp16 \
            --ws-encoding-start-id 10 \
            --ws-encoding-length 10 \
            --make-vocab-size-divisible-by 52224 \
            --seq-length 2048
            """
print("MODEL_ARGS = " + os.environ["MODEL_ARGS"])


# nccl options
os.environ["NCCL_DEBUG"] = "warn"
os.environ["NCCL_IB_DISABLE"] = "0"
os.environ["NCCL_IB_GID_INDEX"] = "3"

# que sentido tiene esto?
#os.environ["PATH"] = os.environ["PATH"]
#os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
os.environ["CWD"] = SCRIPT_DIR

print("ZMQ_ADDR = "+ ZMQ_ADDR)

NUM_SAMPLES=1
MICRO_BSZ=1
WORLD_SIZE=1
TEMP=0.8
TOPP=0.95
SEED=42
DATASET="humaneval"
TODAY:str = datetime.now().strftime('%H_%M_%d_%m_%Y')
rnd:int = random.randint(0,1000)
CHANNEL_PORT = str(rnd + 5000)
MASTER_PORT = str(rnd + 8000)

# save log file

LOG_DIR=os.path.join(MAIN_DIR,"log")
LOG_DIR.encode
os.makedirs(LOG_DIR,exist_ok=True)
LOG_PATH = LOG_DIR + "/" + TODAY + "-generation.log"

INPUT_PATH = MAIN_DIR + "/codegeex/benchmark/humaneval-x/" + LANGUAGE + "/data/humaneval_" + LANGUAGE + ".jsonl.gz"

DATA_DIR=os.path.join(MAIN_DIR,"codegeex/benchmark/humaneval-x/" + LANGUAGE + "/data/humaneval_" + LANGUAGE + ".jsonl.gz")
print("DATA_DIR = " + DATA_DIR)

TMP_DIR = os.path.join(MAIN_DIR, "/codegeex/benchmark/humaneval-x/")

JOB_ID = "codegeex-ns" + str(NUM_SAMPLES) + "-t" + str(TEMP) + "-topp" + str(TOPP) + "-seed" + str(SEED) + "-" +LANGUAGE

#Debugging inputs
LANGUAGE = "java" 

initialize(hostfile = HOSTLIST,
          channel_port = None,
          problem_split = None,
          load_deepspeed = None,
          channel_ip = ZMQ_ADDR, 
          master_port = MASTER_PORT, 
          tokenizer_path = TOKENIZER_PATH,
          temperature = TEMP,
          top_p = TOPP,
          out_seq_length = 1024,
          micro_batch_size = MICRO_BSZ,
          sample_per_problem = NUM_SAMPLES,
          language_type = LANGUAGE,
          dataset = DATASET,
          input_path = INPUT_PATH,
          output_prefix = OUTPUT_PATH + JOB_ID,
          gen_node_world_size = WORLD_SIZE,
          seed = SEED)

"""         
RUN_CMD="python \
  $MAIN_DIR/codegeex/benchmark/humaneval-x/generate_humaneval_x.py \
  --hostfile $HOSTLIST \
  --channel-ip $ZMQ_ADDR \
  --channel-port $CHANNEL_PORT \
  --master-port $MASTER_PORT \
  --tokenizer-path $TOKENIZER_PATH \
  --load-deepspeed \
  --temperature $TEMP \
  --top-p $TOPP \
  --out-seq-length 1024 \
  --micro-batch-size $MICRO_BSZ \
  --samples-per-problem $NUM_SAMPLES \
  --language-type $LANGUAGE \
  --dataset $DATASET \
  --input-path $INPUT_PATH \
  --output-prefix $OUTPUT_PATH/$JOB_ID \
  --gen-node-world-size $WORLD_SIZE \
  --seed $SEED \
  $MODEL_ARGS"

  """

#RUN_CMD="$OPTIONS_NCCL; $OPTIONS_PATH; $RUN_CMD"
#RUN_CMD="cd $CWD; $RUN_CMD"

"""
if (( WORLD_SIZE != 1 )); then
  RUN_CMD="pdsh -R ssh -w ^$HOSTLIST \"$RUN_CMD\""
fi
"""

#echo "$RUN_CMD"
#echo "Writing log to $LOG_PATH"
#eval "$RUN_CMD" > "$LOG_PATH"
#bash $MAIN_DIR/scripts/gather_output.sh $OUTPUT_PATH $JOB_ID 1

