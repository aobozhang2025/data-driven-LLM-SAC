# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (
    DatasetInfoHook,
    EvaluateChatHook,
    VarlenAttnArgsToMessageHubHook,
)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = "/data/home/scyb178/models/baichuan-inc/Baichuan2-7B-Base"
use_varlen_attn = False

# Data
# alpaca_en_path = "tatsu-lab/alpaca"
data_dir = "/data/home/scyb178/data/20250727"
prompt_template = PROMPT_TEMPLATE.default
max_length = 2048
pack_to_max_length = True

# Scheduler & Optimizer
batch_size = 50  # per_device
accumulative_counts = 4
dataloader_num_workers = 0
max_epochs = 400
optim_type = AdamW
lr = 5e-5
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 200
save_total_limit = 8  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 200
SYSTEM = SYSTEM_TEMPLATE.alpaca
evaluation_inputs = [
    'synthesis of single atom Mo supported on N-doped carbon, denoted by Mo1-NC',
    'synthesis of single atom Fe supported on N-doped carbon, denoted by FeNC',
    'preparation of Co doped by N supported on Carbon, denoted by Co-N-C',
    'Synthesis of the single atom Pt supported on N-doped carbon nanotubes',
    'Synthesis of single atom Fe supported on N-doped carbon, denoted by Fe/NC',
    'Synthesis of single atom Co doped by 3 N and 1 P supported on N-C, denoted by Co1-N3P1.',
    'synthesis of single atom Co doped by N supported on graphene',
    'preparation of Mn doped by N supported on Carbon, denoted by Mn-N-C',
    'Preparation of single atom Fe doped by 4 N supported on N-doped Carbon',
    'synthesis of Ni single atom supported on N-doped carbon',
    'synthesis of single atom Ir supported on GO-NH2',
    'synthesis of single atom Ni doped by 4 N supported by carbon with silica sphere template',
    'synthesis of single atom Fe supported on BLG (β-lactoglobulin)',
    'preparation of single atom Ni supported on N-doped carbon',
    'Synthesis of single atom Rh supported on pMOF, denoted by Rh1/pMOF',
    'Preparation of single atom Fe doped by 4 N supported on carbon, denoted by Fe SAs/NC.',
    'synthesis of single atom Fe supported on carbon nanotubes, denoted by Fe-CNTs',
    'synthesis of single atom Co supported on N, S-doped carbon',
    'synthesis of single Fe incorporated on N-rich carbon',
    'Synthesis of single atom Y doped by S and N supported on Carbon, denoted by Y SAs/NC-S',
    'synthesis of single site Pd supported on C',
    'Synthesis of Pt single atom supported on TiO2',
    'synthesis of 0.08% single atom Pt supported on Fe-O',
    'synthesis of single atom Pt supported on dealuminated zeolite Beta containing Fe, denoted by Pt/Fe-DeAlBEA',
    'Synthesis of single atom Pd supported on CeO2.',
    'synthesis of single atom Rh supported on CeO2 nanorod',
    'Preparation of single atom Rh supported on TiO2',
    'Synthesis of single atom Rh supported on CeO2.',
    'Synthesis of single atom metal Mn tetrahedron coordinated supported on ZnO'
]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    ),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
alpaca_en = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='json', data_dir=data_dir),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=alpaca_en,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template,
    ),
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
visualizer = None

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
