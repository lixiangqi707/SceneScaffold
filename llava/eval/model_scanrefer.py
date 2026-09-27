import argparse
import csv
import json
import math
import os
import pathlib
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from llava.conversation import conv_templates
from llava.eval.grounding_box_utils import (
    json_box,
    mask_to_aabb,
    normalize_prediction_masks,
    scanrefer_mask_to_box,
)
from llava.mm_utils import get_model_name_from_path, tokenizer_special_token
from llava.model.builder import load_pretrained_model
from llava.pc_utils import Compose, referseg_transform_eval
from llava.train.train import DataCollatorForSupervisedDataset
from llava.utils import disable_torch_init
from pointgroup_ops import voxelization_idx


templates = [
    "<image>\n Please output the segmentation mask according to the following description. \n{description}"
]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def ponder_collate_fn(batch, max_point=-1):
    """Collate point-cloud fields while preserving point offsets."""
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if max_point > 0:
        accum_num_points = 0
        ret_batches = []
        for data in batch:
            num_coords = data["coord"].shape[0]
            if accum_num_points + num_coords > max_point:
                continue
            accum_num_points += num_coords
            ret_batches.append(data)
        return ponder_collate_fn(ret_batches)

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    if isinstance(batch[0], str):
        return list(batch)
    if isinstance(batch[0], Sequence) and not isinstance(batch[0], (str, bytes)):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [ponder_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    if isinstance(batch[0], Mapping):
        batch = {key: ponder_collate_fn([data[key] for data in batch]) for key in batch[0]}
        for key in batch:
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch

    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        pointcloud_tower_name=args.pointcloud_tower_name,
    )

    if "json" in args.question_file:
        with open(args.question_file, encoding="utf-8") as handle:
            questions = json.load(handle)
    elif "csv" in args.question_file:
        with open(args.question_file, encoding="utf-8") as handle:
            questions = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported question file: {args.question_file}")

    if args.num_chunks < 1 or not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError(f"Invalid chunk selection: {args.chunk_idx}/{args.num_chunks}")
    chunk_size = math.ceil(len(questions) / args.num_chunks)
    source_offset = args.chunk_idx * chunk_size
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file) or ".", exist_ok=True)
    with open(answers_file, "w", encoding="utf-8") as ans_file:
        for idx, source in enumerate(tqdm(questions)):
            scan_file = source["scene_id"]
            scan_data_path = pathlib.Path(args.scan_folder) / f"{scan_file}.pth"
            superpoint_path = pathlib.Path(args.scan_folder) / "../super_points" / f"{scan_file}.bin"

            raw_data = torch.load(scan_data_path, map_location="cpu")
            raw_coord = np.asarray(raw_data["coord"], dtype=np.float32).copy()
            coord = raw_data["coord"].clone() if torch.is_tensor(raw_data["coord"]) else np.array(raw_data["coord"], copy=True)
            instance = np.asarray(raw_data["instance_gt"])
            object_id = int(source["object_id"])
            gt_mask = (instance == object_id)
            color = raw_data["color"]
            superpoint_mask = np.fromfile(superpoint_path, dtype=np.int64)

            pc_data_dict = Compose(referseg_transform_eval)({
                "coord": coord,
                "color": color,
                "superpoint_mask": superpoint_mask,
            })
            grid_coord = pc_data_dict["grid_coord"]
            pc_data_dict["grid_coord"] = torch.cat(
                [torch.zeros((grid_coord.shape[0], 1), dtype=torch.long), torch.as_tensor(grid_coord)],
                dim=1,
            )
            grid_coords = pc_data_dict["grid_coord"]
            spatial_shape = np.clip((grid_coords.max(0)[0][1:] + 1).numpy(), 128, None)
            voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coords, 1, 4)

            for key in pc_data_dict:
                if key in ["coord", "grid_coord", "feat", "offset"]:
                    pc_data_dict[key] = ponder_collate_fn([pc_data_dict[key]])

            description = source.get("description", source.get("utterance"))
            qs = templates[0].format(description=description)
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            device = model.device
            input_ids = tokenizer_special_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
            coord = pc_data_dict["coord"].to(device, dtype=torch.bfloat16)
            voxel_coords = voxel_coords.to(device)
            offset = pc_data_dict["offset"].to(device)
            feat = pc_data_dict["feat"].to(device, dtype=torch.bfloat16)
            p2v_map = p2v_map.to(device)
            v2p_map = v2p_map.to(device)
            superpoint_mask = [torch.tensor(superpoint_mask).to(device)]

            with torch.inference_mode():
                pred_mask = model.generate(
                    input_ids,
                    coord=coord,
                    grid_coord=voxel_coords,
                    offset=offset,
                    feat=feat,
                    p2v_map=p2v_map,
                    v2p_map=v2p_map,
                    spatial_shape=spatial_shape,
                    superpoint_mask=superpoint_mask,
                    conditions=[pc_data_dict["condition"]],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=64,
                    tokenizer=tokenizer,
                    click_mask=[[]],
                    use_cache=True,
                )

            pred_masks = normalize_prediction_masks(pred_mask.cpu().numpy(), len(raw_coord))
            pred_mask = pred_masks[0] if pred_masks else np.zeros(len(raw_coord), dtype=bool)
            gt_mask = gt_mask.astype(bool)
            gt_box = mask_to_aabb(raw_coord, gt_mask)
            pred_box = scanrefer_mask_to_box(
                raw_coord,
                pred_mask,
                eps=args.box_dbscan_eps,
                min_samples=args.box_dbscan_min_samples,
            )
            from llava.eval.grounding_box_utils import aabb_iou
            box_iou = aabb_iou(pred_box, gt_box)

            intersection = np.sum(np.logical_and(pred_mask, gt_mask))
            union = np.sum(np.logical_or(pred_mask, gt_mask))
            iou = float(intersection / union) if union else 0.0
            global_question_id = source_offset + idx
            ans_file.write(json.dumps({
                "scene_id": scan_file,
                "object_id": object_id,
                "ann_id": int(source.get("ann_id", -1)),
                "question_id": global_question_id,
                "prompt": qs,
                "model_id": model_name,
                "iou": iou,
                "tp50": int(iou >= 0.5),
                "tp25": int(iou >= 0.25),
                "box_iou": float(box_iou),
                "box_tp50": int(box_iou >= 0.5),
                "box_tp25": int(box_iou >= 0.25),
                "pred_box": json_box(pred_box),
                "gt_box": json_box(gt_box),
                "num_pred_points": int(pred_mask.sum()),
                "num_pred_masks": len(pred_masks),
            }) + "\n")
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--pointcloud-tower-name", type=str, default=None)
    parser.add_argument("--scan-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--data_version", type=str, default="v0")
    parser.add_argument("--box-dbscan-eps", type=float, default=0.08)
    parser.add_argument("--box-dbscan-min-samples", type=int, default=10)
    eval_model(parser.parse_args())
