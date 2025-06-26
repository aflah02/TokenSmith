#! /bin/bash

source /NS/venvs/work/afkhan/neox_updated_env/bin/activate

# echo python env and path
echo "Python version: $(python --version)"
echo "Python executable: $(which python)"
echo "Current working directory: $(pwd)"

NEOX_DIR="/NS/llm-pretraining/work/afkhan/USC_Colab/gpt-neox"

log_file="/NS/llm-pretraining/work/afkhan/tokensmith/artifacts/tokenize.log"

# Using https://huggingface.co/EleutherAI/gpt-neox-20b/blob/main/tokenizer.json 

python $NEOX_DIR/tools/datasets/preprocess_data.py \
                  --input /NS/llm-pretraining/work/afkhan/tokensmith/artifacts/data.jsonl \
                  --output-prefix /NS/llm-pretraining/work/afkhan/tokensmith/artifacts/data_tokenized \
                  --vocab /NS/llm-pretraining/work/afkhan/tokensmith/artifacts/tokenizer.json \
                  --dataset-impl mmap \
                  --tokenizer-type HFTokenizer \
                  --append-eod \
                  --workers 1 2>&1 | tee ${log_file}
echo "Tokenization completed. Log file: ${log_file}"