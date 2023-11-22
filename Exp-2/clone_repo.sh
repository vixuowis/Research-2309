#!/bin/bash

cd ~/autodl-tmp

model_name="CodeLlama-13b-Python-hf"
model_id="codellama/${model_name}"

git clone git@hf.co:${model_id}

cd ${model_name}

git remote set-url origin https://Vixuowis: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@huggingface.co/${model_id}

git pull origin

git lfs pull