from typing import *
import shutil
from transformers import pipeline

def copy_pipeline():
    '''
    This function copies the file where `PairClassificationPipeline` is defined inside the folder `"test-dynamic-pipeline"`,
    along with saving the model and tokenizer of the pipeline, before pushing everything into the repository
    `{your_username}/test-dynamic-pipeline`. After that, anyone can use it as long as they provide the option
    `trust_remote_code=True`:
    '''
    shutil.copyfile('path/to/PairClassificationPipeline.py', 'test-dynamic-pipeline/PairClassificationPipeline.py')

    model = 'path/to/model'
    tokenizer = 'path/to/tokenizer'

    pipeline.save_pretrained('{your_username}/test-dynamic-pipeline', model=model, tokenizer=tokenizer)
