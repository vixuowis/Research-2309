from f00049_hf_hub_download import *
import os
import shutil


def test_hf_hub_download():
    # Test case 1
    repo_id = "bigscience/T0_3B"
    filename = "config.json"
    cache_dir = "./your/path/bigscience_t0"
    hf_hub_download(repo_id, filename, cache_dir)
    assert os.path.exists(os.path.join(cache_dir, filename))

    # Test case 2
    repo_id = "owner/repo_name"
    filename = "file.txt"
    cache_dir = "./your/path"
    hf_hub_download(repo_id, filename, cache_dir)
    assert os.path.exists(os.path.join(cache_dir, filename))

    # Test case 3
    repo_id = "huggingface/datasets"
    filename = "dataset.py"
    cache_dir = "./your/path/datasets"
    hf_hub_download(repo_id, filename, cache_dir)
    assert os.path.exists(os.path.join(cache_dir, filename))

    # Test case 4
    repo_id = "huggingface/transformers"
    filename = "modeling_utils.py"
    cache_dir = "./your/path/transformers"
    hf_hub_download(repo_id, filename, cache_dir)
    assert os.path.exists(os.path.join(cache_dir, filename))

    # Test case 5
    repo_id = "huggingface/tokenizers"
    filename = "tokenization_utils_base.py"
    cache_dir = "./your/path/tokenizers"
    hf_hub_download(repo_id, filename, cache_dir)
    assert os.path.exists(os.path.join(cache_dir, filename))


test_hf_hub_download()

