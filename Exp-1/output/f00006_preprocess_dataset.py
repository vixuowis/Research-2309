from typing import *
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer
import torch
import torchaudio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers.trainer_utils import get_last_checkpoint
from transformers import HfArgumentParser
from transformers.trainer_utils import is_main_process
from transformers.trainer_utils import set_seed
from transformers.trainer_utils import distributed_broadcast_scalars
from transformers.trainer_utils import DistributedTensorGatherer
from transformers.trainer_utils import nested_concat
from transformers.trainer_utils import nested_numpify
from transformers.trainer_utils import nested_xla_mesh_reduce
from transformers.trainer_utils import nested_xla_mesh_reduce
from transformers.trainer_utils import save_metrics
from transformers.trainer_utils import speed_metrics
from transformers.trainer_utils import TrainOutput
from transformers.trainer_utils import PredictionOutput
from transformers.trainer_utils import DistributedTensorGatherer
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_utils import is_main_process
from transformers.trainer_utils import set_seed
from transformers.trainer_utils import distributed_broadcast_scalars
from transformers.trainer_utils import nested_concat
from transformers.trainer_utils import nested_numpify
from transformers.trainer_utils import nested_xla_mesh_reduce
from transformers.trainer_utils import nested_xla_mesh_reduce
from transformers.trainer_utils import save_metrics
from transformers.trainer_utils import speed_metrics
from transformers.trainer_utils import TrainOutput
from transformers.trainer_utils import PredictionOutput


def preprocess_dataset(dataset):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def prepare_dataset(batch):
        batch['input_values'] = processor(batch['speech'], sampling_rate=16000).input_values
        return batch

    dataset = dataset.map(prepare_dataset)
    return dataset
