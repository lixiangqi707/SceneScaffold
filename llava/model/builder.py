#    Modified from LLaVA
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


SSC_SU_TOLERANT_MODULE_KEYS = (
    "pc_ssc_scene_proj",
    "pc_ssc_ent_sem_proj",
)


def _config_is_llava(config):
    if config is None:
        return False
    model_type = str(getattr(config, "model_type", "")).lower()
    if "llava" in model_type:
        return True
    archs = [str(arch).lower() for arch in (getattr(config, "architectures", None) or [])]
    if any("llava" in arch for arch in archs):
        return True
    return "llava" in config.__class__.__name__.lower()


def _infer_llava_checkpoint(model_path, model_base, model_name):
    if "llava" in str(model_name).lower():
        return True
    if "llava" in str(model_base).lower():
        return True
    if "llava" in str(model_path).lower():
        return True

    config_candidates = [model_path]
    if model_base is not None:
        config_candidates.append(model_base)

    for candidate in config_candidates:
        try:
            cfg = AutoConfig.from_pretrained(candidate, trust_remote_code=True)
        except Exception:
            continue
        if _config_is_llava(cfg):
            print(f"Detected LLaVA checkpoint via config inspection: {candidate}")
            return True
    return False


def _print_missing_ssc_su_keys_once(missing_keys):
    if getattr(_print_missing_ssc_su_keys_once, "_printed", False):
        return
    missing_su_keys = sorted(
        {key for key in missing_keys if any(module_key in key for module_key in SSC_SU_TOLERANT_MODULE_KEYS)}
    )
    if len(missing_su_keys) > 0:
        print(f"Missing SSC3D-SU keys from checkpoint, newly initialized this run: {missing_su_keys}")
        _print_missing_ssc_su_keys_once._printed = True


def _drop_mismatched_ssc_role_embed(state_dict, model):
    model_state = model.state_dict()
    filtered = {}
    dropped = []
    for key, value in state_dict.items():
        if (
            key in model_state
            and isinstance(value, torch.Tensor)
            and tuple(value.shape) != tuple(model_state[key].shape)
            and (
                key.endswith("pc_ssc_token_role_embed")
                or any(module_key in key for module_key in SSC_SU_TOLERANT_MODULE_KEYS)
            )
        ):
            dropped.append(f"{key}: checkpoint={tuple(value.shape)}, model={tuple(model_state[key].shape)}")
            continue
        filtered[key] = value
    if len(dropped) > 0:
        print(f"Skipping mismatched SSC3D role embedding from checkpoint: {dropped}")
    return filtered


def _upgrade_legacy_scalar_tsp_params(state_dict):
    # Scalar-shape compatibility for checkpoint loading.
    # Current TSP3D clean version actively uses `pc_rel_seg_alpha`.
    # The remaining keys are legacy-only compatibility for older checkpoints.
    scalar_like_keys = [
        "pc_rel_seg_alpha",
        "pc_ssc_seg_alpha",
        "pc_state_alpha_obj",
        "pc_state_alpha_sup",
        "pc_state_alpha_rel",
        "pc_rel_source_gate_ref",
        "pc_rel_source_gate_str",
        "pc_rel_source_gate_obj",
        "pc_rel_source_gate_sup",
        "pc_state_rel_refine_alpha",
        "pc_state_seg_alpha",
    ]
    upgraded = {}
    for key, value in state_dict.items():
        new_value = value
        for name in scalar_like_keys:
            if key.endswith(name) and isinstance(value, torch.Tensor) and value.ndim == 0:
                new_value = value.unsqueeze(0)
                break
        upgraded[key] = new_value
    return upgraded


def _print_tsp_scalar_shapes_once(state_dict):
    if getattr(_print_tsp_scalar_shapes_once, "_printed", False):
        return
    # Debug print for scalar-shape compatibility checks only.
    # Current TSP3D clean version actively uses `pc_rel_seg_alpha`;
    # the rest are legacy-only fields.
    scalar_like_keys = [
        "pc_rel_seg_alpha",
        "pc_ssc_seg_alpha",
        "pc_state_alpha_obj",
        "pc_state_alpha_sup",
        "pc_state_alpha_rel",
        "pc_rel_source_gate_ref",
        "pc_rel_source_gate_str",
        "pc_rel_source_gate_obj",
        "pc_rel_source_gate_sup",
        "pc_state_rel_refine_alpha",
        "pc_state_seg_alpha",
    ]
    shape_lines = []
    for key, value in state_dict.items():
        for name in scalar_like_keys:
            if key.endswith(name) and isinstance(value, torch.Tensor):
                shape_lines.append(f"{key}: {tuple(value.shape)}")
                break
    if len(shape_lines) > 0:
        print("[TSP3D load scalar-shapes]")
        for line in sorted(shape_lines):
            print(line)
    _print_tsp_scalar_shapes_once._printed = True


def _maybe_initialize_pointcloud_state_modules(model):
    if not hasattr(model, "get_pointcloud_tower") or not hasattr(model, "get_model"):
        return
    pointcloud_tower = model.get_pointcloud_tower()
    if pointcloud_tower is None:
        return
    config = getattr(model, "config", None)
    inner_model = model.get_model()
    pc_hidden_size = getattr(config, "pc_hidden_size", None)
    if pc_hidden_size is None:
        pc_hidden_size = getattr(pointcloud_tower, "hidden_size", getattr(config, "mm_hidden_size", 1024))
        if config is not None:
            config.pc_hidden_size = pc_hidden_size
    seg_hidden_size = getattr(config, "ssc_seg_hidden_size", None)
    if seg_hidden_size is None:
        seg_hidden_size = getattr(config, "tsp_seg_hidden_size", None)
    if hasattr(pointcloud_tower, "seg_query_dim"):
        try:
            seg_hidden_size = int(pointcloud_tower.seg_query_dim)
        except Exception:
            pass
    if config is not None and seg_hidden_size is not None:
        config.tsp_seg_hidden_size = seg_hidden_size
        config.ssc_seg_hidden_size = seg_hidden_size
    if hasattr(inner_model, "_maybe_initialize_pc_dvc_modules") and not bool(getattr(config, "tsp_enable", False)):
        inner_model._maybe_initialize_pc_dvc_modules(pc_hidden_size)
    if hasattr(inner_model, "_maybe_initialize_pc_tsp_modules"):
        inner_model._maybe_initialize_pc_tsp_modules(pc_hidden_size)
    if hasattr(inner_model, "_maybe_initialize_pc_ssc_modules"):
        inner_model._maybe_initialize_pc_ssc_modules(pc_hidden_size)


def load_pretrained_model(model_path, model_base, model_name, pointcloud_tower_name=None, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    # E6 exports use the standard PEFT adapter layout even when the directory
    # name does not contain "lora". Detect that format from its files rather
    # than routing it through the legacy projector-only loader.
    has_adapter_export = (
        model_base is not None
        and os.path.isfile(os.path.join(model_path, "adapter_config.json"))
        and os.path.isfile(os.path.join(model_path, "non_lora_trainables.bin"))
        and any(
            os.path.isfile(os.path.join(model_path, filename))
            for filename in ("adapter_model.bin", "adapter_model.safetensors")
        )
    )
    local_files_only = bool(model_base and os.path.isdir(model_base))

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    is_llava_model = _infer_llava_checkpoint(model_path, model_base, model_name)

    if is_llava_model:
        # Load LLaVA model
        if ('lora' in model_name.lower() or has_adapter_export) and model_base is None:
            warnings.warn('There is a LoRA adapter export but no `model_base` is provided. Please provide the `model_base` argument.')
        if ('lora' in model_name.lower() or has_adapter_export) and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path, local_files_only=local_files_only)

            if os.path.exists(os.path.join(model_path, 'tokenizer.model')):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=local_files_only)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=local_files_only)

            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=lora_cfg_pretrained,
                local_files_only=local_files_only,
                **kwargs
            )
            token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features

            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))

            pointcloud_tower = model.get_pointcloud_tower()
            if not pointcloud_tower.is_loaded:
                pointcloud_tower.load_model(pointcloud_tower_name=pointcloud_tower_name)
                pointcloud_tower.to(device=model.device, dtype=torch.bfloat16)
            _maybe_initialize_pointcloud_state_modules(model)

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            non_lora_trainables = _upgrade_legacy_scalar_tsp_params(non_lora_trainables)
            non_lora_trainables = _drop_mismatched_ssc_role_embed(non_lora_trainables, model)
            _print_tsp_scalar_shapes_once(non_lora_trainables)
            missing_keys, unexpected_keys = model.load_state_dict(non_lora_trainables, strict=False)
            _print_missing_ssc_su_keys_once(missing_keys)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')

        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, local_files_only=local_files_only)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_files_only)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, local_files_only=local_files_only, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=local_files_only)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, local_files_only=local_files_only, **kwargs)

            if os.path.exists(os.path.join(model_path, 'tokenizer.model')):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=local_files_only)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=local_files_only)
                
            pointcloud_tower = model.get_pointcloud_tower()
            if not pointcloud_tower.is_loaded:
                pointcloud_tower.load_model(pointcloud_tower_name=pointcloud_tower_name)
                pointcloud_tower.to(device=model.device, dtype=torch.bfloat16)
            _maybe_initialize_pointcloud_state_modules(model)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.bfloat16) for k, v in mm_projector_weights.items()}
            mm_projector_weights = _drop_mismatched_ssc_role_embed(mm_projector_weights, model)
            missed, unexp = model.load_state_dict(mm_projector_weights, strict=False)
            _print_missing_ssc_su_keys_once(missed)

        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
