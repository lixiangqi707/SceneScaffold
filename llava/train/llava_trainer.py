import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import Dict, List, Optional
import deepspeed
from collections import defaultdict
from itertools import cycle
import random


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def _normalize_scalar_tsp_params_for_save(state_dict):
    scalar_like_keys = [
        # Current TSP3D clean version scalar.
        "pc_rel_seg_alpha",
        "pc_ssc_seg_alpha",
        # Legacy-only compatibility: older checkpoints used this name.
        "pc_state_seg_alpha",
    ]
    normalized = {}
    for key, value in state_dict.items():
        new_value = value
        for name in scalar_like_keys:
            if key.endswith(name) and isinstance(value, torch.Tensor) and value.ndim == 0:
                new_value = value.unsqueeze(0)
                break
        normalized[key] = new_value
    return normalized


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

def get_task_length_grouped_indices(lengths, batch_size, world_size, generator=None):

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    from collections import defaultdict
    task_indices, task_lengths = defaultdict(list), defaultdict(list)
    for i, (task_id, l) in enumerate(lengths):
        task_indices[task_id].append(i)
        task_lengths[task_id].append(l)
    
    task_ids = list(task_indices.keys())
    task_shuffle = {}
    for task_id in task_ids:
        task_shuffle[task_id] = [task_indices[task_id][i] for i in get_length_grouped_indices(task_lengths[task_id], batch_size, world_size, generator=None)]

    megabatch_size = world_size * batch_size
    task_megabatches = {}
    for task_id in task_ids:
        task_megabatches[task_id] = [task_shuffle[task_id][i: i + megabatch_size] for i in range(0, len(task_shuffle[task_id]), megabatch_size)]

    megabatches = []
    for task_id in task_ids:
        megabatches.extend(task_megabatches[task_id][:-1])
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    return [i for megabatch in megabatches for i in megabatch]

def split_to_batches(indices, lengths, batch_size):
    if isinstance(lengths, list):
        length_dict = {i: l for i, l in zip(indices, lengths)}
    elif isinstance(lengths, dict):
        length_dict = lengths
    else:
        raise ValueError("Unsupported lengths format")

    return [sorted(indices[i : i + batch_size], key=lambda idx: length_dict[idx], reverse=True) 
            for i in range(0, len(indices), batch_size)]


def get_per_batch_task_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    重新实现：每个 batch 内只有一个任务，而每个 mega batch (batch_size * world_size) 内可能混合多个任务。
    """
    assert all(l != 0 for _, l in lengths), "Should not have zero length."
    
    task_indices, task_lengths = defaultdict(list), defaultdict(list)
    for i, (task_id, l) in enumerate(lengths):
        task_indices[task_id].append(i)
        task_lengths[task_id].append(l)

    task_ids = list(task_indices.keys())
    task_batches = []

    # 先对每个任务内部进行排序并切成 batch_size 大小
    for task_id in task_ids:
        shuffled_indices = torch.randperm(len(task_indices[task_id]), generator=generator).tolist()
        task_indices[task_id] = [task_indices[task_id][i] for i in shuffled_indices]
        task_batches.extend(split_to_batches(task_indices[task_id], task_lengths[task_id], batch_size))

    # 任务级别的 batch pool
    random.shuffle(task_batches)  # 打乱任务顺序
    
    # 组织 mega batch，每个 mega batch 内的 batch 可能来自不同任务
    megabatches = []
    for i in range(0, len(task_batches), world_size):
        megabatch = task_batches[i : i + world_size]
        if len(megabatch) == world_size:
            megabatches.append(megabatch)

    # 最后打乱 mega batch 顺序
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_task_ratio_grouped_indices(lengths, batch_size, world_size, task_ratios={0: 0.3, 1: 0.3, 2: 0.4 }, generator=None):
    assert all(l != 0 for _, l in lengths), "Should not have zero length."
    from collections import defaultdict

    task_indices, task_lengths = defaultdict(list), defaultdict(list)
    for i, (task_id, l) in enumerate(lengths):
        task_indices[task_id].append(i)
        task_lengths[task_id].append(l)
    
    task_ids = list(task_indices.keys())
    
    task_shuffle = {}
    for task_id in task_ids:
        task_shuffle[task_id] = [
            task_indices[task_id][i] for i in get_length_grouped_indices(
                task_lengths[task_id], batch_size, world_size, generator=None
            )
        ]
    
    megabatch_size = world_size * batch_size
    task_sample_counts = {
        task_id: int(megabatch_size * task_ratios.get(task_id, 1 / len(task_ids))) 
        for task_id in task_ids
    }
    
    total_samples = sum(task_sample_counts.values())
    diff = megabatch_size - total_samples
    if diff != 0:
        max_task = max(task_sample_counts, key=lambda x: task_sample_counts[x])
        task_sample_counts[max_task] += diff

    task_megabatches = {task_id: [] for task_id in task_ids}
    for task_id in task_ids:
        task_list = task_shuffle[task_id]
        task_megabatches[task_id] = [task_list[i: i + task_sample_counts[task_id]] 
                                     for i in range(0, len(task_list), task_sample_counts[task_id])]
    max_megabatches = max(len(mb) for mb in task_megabatches.values())

    for task_id in task_ids:
        if len(task_megabatches[task_id]) < max_megabatches:
            extra_batches_needed = max_megabatches - len(task_megabatches[task_id])
            for _ in range(extra_batches_needed):
                # Randomly pick an existing batch to duplicate
                if task_megabatches[task_id]:
                    duplicate_batch = random.choice(task_megabatches[task_id])
                    task_megabatches[task_id].append(duplicate_batch)

    megabatches = []
    while any(task_megabatches.values()):
        megabatch = []
        for task_id in task_ids:
            if task_megabatches[task_id]: 
                megabatch.extend(task_megabatches[task_id].pop(0)) 
        megabatches.append(megabatch)

    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
        group_by_task: bool=False,
        group_by_task_ratio: bool=False,
        group_by_task_per_batch: bool=False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto
        self.group_by_task = group_by_task
        self.group_by_task_ratio = group_by_task_ratio
        self.group_by_task_per_batch = group_by_task_per_batch

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_task_ratio:
            indices = get_task_ratio_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        elif self.group_by_task_per_batch:
            indices = get_per_batch_task_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        elif self.group_by_task:
            indices = get_task_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        elif self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for name, param in self.model.get_model().named_parameters():
            if 'pointcloud_tower' in name or 'pointcloud_decoder' in name:
                deepspeed.zero.register_external_parameter(self.model, param)
        self._dvc_last_runtime_step = None
        self._dvc_last_logged_phase = None
        self._dvc_schedule_reported = False

    def _get_dvc_specific_keywords(self):
        return [
            "pc_layout_adapter",
            "pc_entity_tail_role_head",
            "pc_layout_role_head",
            "pc_layout_mix_gate",
            "pc_layout_slot_gate",
            "pc_ssc_entity_primitive_head",
            "pc_ssc_state_query_ent",
            "pc_ssc_state_query_frame",
            "pc_ssc_state_query_rel",
            "pc_ssc_state_ent_block",
            "pc_ssc_state_frame_block",
            "pc_ssc_state_rel_block",
            "pc_ssc_token_role_embed",
            "pc_ssc_seg_reader",
            "pc_ssc_seg_alpha",
            "pc_ssc_scene_proj",
            "pc_ssc_ent_sem_proj",
            "pc_ssc_rel_boundary_head",
            "pc_ssc_rel_region_head",
        ]

    def _get_tsp_specific_keywords(self):
        return [
            "pc_state_query_ent",
            "pc_state_query_frame",
            "pc_state_query_rel",
            "pc_state_ent_block",
            "pc_state_frame_block",
            "pc_state_rel_block",
            "pc_rel_to_seg",
            "pc_rel_seg_alpha",
            "pc_role_embed",
        ]

    def _match_any_keyword(self, name, keywords):
        return any(keyword in name for keyword in keywords)

    def _iter_runtime_models(self, model):
        stack = [model, self.model]
        visited = set()
        while len(stack) > 0:
            current = stack.pop()
            if current is None or id(current) in visited:
                continue
            visited.add(id(current))
            yield current

            if hasattr(current, "module"):
                stack.append(current.module)
            if hasattr(current, "model") and current.model is not current:
                stack.append(current.model)

            get_model_fn = getattr(current, "get_model", None)
            if callable(get_model_fn):
                try:
                    inner_model = get_model_fn()
                except Exception:
                    inner_model = None
                if inner_model is not None and inner_model is not current:
                    stack.append(inner_model)

    def _set_dvc_runtime_state(self, model, phase, gate_scale):
        for runtime_model in self._iter_runtime_models(model):
            if hasattr(runtime_model, "config"):
                runtime_model.config.current_dvc_phase = phase
                runtime_model.config.current_dvc_gate_scale = gate_scale

    def _mean_scalar_from_debug(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return 0.0
            vals = [float(v) for v in value]
            return sum(vals) / float(len(vals))
        try:
            return float(value)
        except Exception:
            return 0.0

    def _collect_dvc_debug_logs(self):
        logs = {}
        for runtime_model in self._iter_runtime_models(self.model):
            debug = getattr(runtime_model, "_last_dvc_debug", None)
            if isinstance(debug, dict):
                frame_counts = debug.get("frame_token_counts", debug.get("layout_token_counts", 0.0))
                logs = {
                    "dvc/entity_token_count_mean": self._mean_scalar_from_debug(debug.get("entity_token_counts", 0.0)),
                    "dvc/frame_token_count_mean": self._mean_scalar_from_debug(frame_counts),
                    "dvc/boundary_selected_count_mean": self._mean_scalar_from_debug(debug.get("boundary_selected_counts", 0.0)),
                    "dvc/region_selected_count_mean": self._mean_scalar_from_debug(debug.get("region_selected_counts", 0.0)),
                    "dvc/final_token_count_mean": self._mean_scalar_from_debug(debug.get("final_token_counts", 0.0)),
                    "dvc/final_token_count_match": self._mean_scalar_from_debug(debug.get("final_token_count_match", 0.0)),
                }
                break
        return logs

    def _collect_tsp_debug_logs(self):
        logs = {}
        for runtime_model in self._iter_runtime_models(self.model):
            debug = getattr(runtime_model, "_last_tsp_debug", None)
            if isinstance(debug, dict):
                logs = {
                    "tsp/ent_state_norm_mean": self._mean_scalar_from_debug(debug.get("ent_state_norm_mean", 0.0)),
                    "tsp/frame_state_norm_mean": self._mean_scalar_from_debug(debug.get("frame_state_norm_mean", 0.0)),
                    "tsp/rel_state_norm_mean": self._mean_scalar_from_debug(debug.get("rel_state_norm_mean", 0.0)),
                    "tsp/boundary_selected_count_mean": self._mean_scalar_from_debug(debug.get("boundary_selected_counts", 0.0)),
                    "tsp/region_selected_count_mean": self._mean_scalar_from_debug(debug.get("region_selected_counts", 0.0)),
                    "tsp/rel_seg_alpha": self._mean_scalar_from_debug(debug.get("tsp_rel_seg_alpha", 0.0)),
                    "tsp/final_token_count_match": self._mean_scalar_from_debug(debug.get("final_token_count_match", 0.0)),
                }
                break
        return logs

    def _collect_ssc_debug_logs(self):
        logs = {}
        for runtime_model in self._iter_runtime_models(self.model):
            debug = getattr(runtime_model, "_last_ssc_debug", None)
            if isinstance(debug, dict):
                logs = {
                    "ssc/ent_state_norm_mean": self._mean_scalar_from_debug(debug.get("ent_state_norm_mean", 0.0)),
                    "ssc/frame_state_norm_mean": self._mean_scalar_from_debug(debug.get("frame_state_norm_mean", 0.0)),
                    "ssc/rel_state_norm_mean": self._mean_scalar_from_debug(debug.get("rel_state_norm_mean", 0.0)),
                    "ssc/frame_valid_count_mean": self._mean_scalar_from_debug(debug.get("frame_valid_counts", 0.0)),
                    "ssc/frame_cov_loss": self._mean_scalar_from_debug(debug.get("frame_cov_loss", 0.0)),
                    "ssc/rel_boundary_loss": self._mean_scalar_from_debug(debug.get("rel_boundary_loss", 0.0)),
                    "ssc/rel_region_loss": self._mean_scalar_from_debug(debug.get("rel_region_loss", 0.0)),
                    "ssc/final_token_count_match": self._mean_scalar_from_debug(debug.get("final_token_count_match", 0.0)),
                }
                break
        return logs

    def _maybe_update_dvc_schedule(self, model=None, force=False):
        runtime_model = model if model is not None else self.model
        config = getattr(runtime_model, "config", getattr(self.model, "config", None))
        if config is None or not getattr(config, "dvc_enable", False):
            return
        if getattr(config, "dvc_phase_schedule", "step") != "step":
            return

        total_steps = int(getattr(self.state, "max_steps", 0) or getattr(self.args, "max_steps", 0) or 0)
        if total_steps <= 0:
            return

        current_step = int(getattr(self.state, "global_step", 0))
        if not force and self._dvc_last_runtime_step == current_step:
            return

        warmup_ratio = float(getattr(config, "dvc_warmup_ratio", 0.1))
        warmup_ratio = max(0.0, min(warmup_ratio, 1.0))
        step_ratio = float(current_step) / float(max(total_steps, 1))

        if warmup_ratio > 0.0 and step_ratio < warmup_ratio:
            current_phase = "dvc_warmup"
            warmup_progress = step_ratio / warmup_ratio
            dvc_lr_scale = float(getattr(config, "dvc_warmup_lr_scale", 0.1))
            gate_scale = warmup_progress if getattr(config, "dvc_zero_perturbation", True) else 1.0
        else:
            current_phase = "dvc_joint_finetune"
            warmup_progress = 1.0
            dvc_lr_scale = float(getattr(config, "dvc_joint_lr_scale", 1.0))
            gate_scale = 1.0

        self._set_dvc_runtime_state(runtime_model, current_phase, gate_scale)

        if self.optimizer is not None:
            baseline_lr_refs = {}
            for group in self.optimizer.param_groups:
                if group.get("dvc_group_role") == "baseline_stage2":
                    baseline_lr_refs[group.get("dvc_lr_ref_key")] = group["lr"]
            for group in self.optimizer.param_groups:
                if group.get("dvc_group_role") == "dvc_specific":
                    reference_lr = baseline_lr_refs.get(group.get("dvc_lr_ref_key"), group["lr"])
                    group["lr"] = reference_lr * dvc_lr_scale

        if not self._dvc_schedule_reported:
            warmup_steps = int(total_steps * warmup_ratio)
            joint_steps = max(total_steps - warmup_steps, 0)
            logger.info(
                "DVC phase schedule: "
                f"total_steps={total_steps}, warmup_ratio={warmup_ratio:.4f}, "
                f"warmup_steps={warmup_steps}, joint_finetune_steps={joint_steps}"
            )
            self._dvc_schedule_reported = True

        if force or current_phase != self._dvc_last_logged_phase:
            logger.info(
                "DVC phase update: "
                f"phase={current_phase}, step={current_step}/{total_steps}, "
                f"step_ratio={step_ratio:.6f}, warmup_progress={warmup_progress:.6f}, "
                f"dvc_lr_scale={dvc_lr_scale:.6f}, gate_scale={gate_scale:.6f}"
            )
            self._dvc_last_logged_phase = current_phase

        self._dvc_last_runtime_step = current_step
        

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_task_ratio:
            lengths = self.train_dataset.task_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_task_ratio=True
            )
        elif self.args.group_by_task_length_per_batch:
            lengths = self.train_dataset.task_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_task_per_batch=True
            )
        elif self.args.group_by_task_length:
            lengths = self.train_dataset.task_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_task=True
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            if getattr(opt_model.config, "dvc_enable", False) and getattr(opt_model.config, "dvc_phase_schedule", "step") == "step":
                dvc_specific_keywords = self._get_dvc_specific_keywords()

                def resolve_base_lr(name):
                    if self.args.mm_projector_lr is not None and "mm_projector" in name:
                        return self.args.mm_projector_lr
                    if self.args.pointcloud_tower_lr is not None and "pointcloud_tower" in name:
                        return self.args.pointcloud_tower_lr
                    if self.args.pc_projector_lr is not None and "pc_projector" in name:
                        return self.args.pc_projector_lr
                    if self.args.pointcloud_decoder_lr is not None and "pointcloud_decoder" in name:
                        return self.args.pointcloud_decoder_lr
                    return self.args.learning_rate

                optimizer_grouped_parameters = {}
                for name, param in opt_model.named_parameters():
                    if not param.requires_grad:
                        continue
                    group_role = "dvc_specific" if self._match_any_keyword(name, dvc_specific_keywords) else "baseline_stage2"
                    use_decay = name in decay_parameters
                    base_lr = resolve_base_lr(name)
                    group_key = (group_role, use_decay, float(base_lr))
                    if group_key not in optimizer_grouped_parameters:
                        optimizer_grouped_parameters[group_key] = {
                            "params": [],
                            "weight_decay": self.args.weight_decay if use_decay else 0.0,
                            "lr": base_lr,
                            "dvc_group_role": group_role,
                            "dvc_base_lr": float(base_lr),
                            "dvc_lr_ref_key": (float(base_lr), bool(use_decay)),
                        }
                    optimizer_grouped_parameters[group_key]["params"].append(param)
                optimizer_grouped_parameters = list(optimizer_grouped_parameters.values())
            elif any([self.args.mm_projector_lr, self.args.pointcloud_tower_lr, self.args.pc_projector_lr, self.args.pointcloud_decoder_lr]):
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                tower_parameters = [name for name, _ in opt_model.named_parameters() if "pointcloud_tower" in name]
                pc_projector_parameters = [name for name, _ in opt_model.named_parameters() if "pc_projector" in name]
                decoder_parameters = [name for name, _ in opt_model.named_parameters() if "pointcloud_decoder" in name]
                
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in tower_parameters and n not in pc_projector_parameters and n not in decoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in tower_parameters and n not in pc_projector_parameters and n not in decoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                
                if self.args.mm_projector_lr is not None:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ])
                
                if self.args.pointcloud_tower_lr is not None:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.pointcloud_tower_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in tower_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.pointcloud_tower_lr,
                        },
                    ])
                
                if self.args.pc_projector_lr is not None:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in pc_projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.pc_projector_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in pc_projector_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.pc_projector_lr,
                        },
                    ])
                
                if self.args.pointcloud_decoder_lr is not None:
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in decoder_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.pointcloud_decoder_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in decoder_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.pointcloud_decoder_lr,
                        },
                    ])

            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def training_step(self, model, inputs):
        self._maybe_update_dvc_schedule(model=model)
        return super().training_step(model, inputs)

    def log(self, logs: Dict[str, float]) -> None:
        dvc_logs = self._collect_dvc_debug_logs()
        tsp_logs = self._collect_tsp_debug_logs()
        ssc_logs = self._collect_ssc_debug_logs()
        if len(dvc_logs) > 0:
            logs = dict(logs)
            logs.update(dvc_logs)
        if len(tsp_logs) > 0:
            logs = dict(logs)
            logs.update(tsp_logs)
        if len(ssc_logs) > 0:
            logs = dict(logs)
            logs.update(ssc_logs)
        super().log(logs)

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = [
                'mm_projector',
                'vision_resampler',
                'lm_head_seg',
                'hidden_seg_fc',
                'alignment_proj',
                'pc_layout_adapter',
                'pc_entity_tail_role_head',
                'pc_layout_role_head',
                'pc_layout_mix_gate',
                'pc_layout_slot_gate',
                'pc_state_query_ent',
                'pc_state_query_frame',
                'pc_state_query_rel',
                'pc_state_ent_block',
                'pc_state_frame_block',
                'pc_state_rel_block',
                'pc_rel_to_seg',
                'pc_rel_seg_alpha',
                'pc_role_embed',
                'pc_ssc_entity_primitive_head',
                'pc_ssc_state_query_ent',
                'pc_ssc_state_query_frame',
                'pc_ssc_state_query_rel',
                'pc_ssc_state_ent_block',
                'pc_ssc_state_frame_block',
                'pc_ssc_state_rel_block',
                'pc_ssc_token_role_embed',
                'pc_ssc_seg_reader',
                'pc_ssc_seg_alpha',
                'pc_ssc_scene_proj',
                'pc_ssc_ent_sem_proj',
                'pc_ssc_rel_boundary_head',
                'pc_ssc_rel_region_head',
            ]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            weight_to_save = _normalize_scalar_tsp_params_for_save(weight_to_save)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
