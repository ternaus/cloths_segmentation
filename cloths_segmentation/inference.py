import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.dl.pytorch.utils import state_dict_from_disk, tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad_to_size, unpad_from_size
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--input_path", type=Path, help="Path with images.", required=True)
    arg("-c", "--config_path", type=Path, help="Path to config.", required=True)
    arg("-o", "--output_path", type=Path, help="Path to save masks.", required=True)
    arg("-b", "--batch_size", type=int, help="batch_size", default=1)
    arg("-j", "--num_workers", type=int, help="num_workers", default=12)
    arg("-w", "--weight_path", type=str, help="Path to weights.", required=True)
    arg("--world_size", default=-1, type=int, help="number of nodes for distributed training")
    arg("--local_rank", default=-1, type=int, help="node rank for distributed training")
    arg("--fp16", action="store_true", help="Use fp6")
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(self, file_paths: List[Path], transform: albu.Compose) -> None:
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        image_path = self.file_paths[idx]

        image = load_rgb(image_path)
        height, width = image.shape[:2]

        image = self.transform(image=image)["image"]
        pad_dict = pad_to_size((max(image.shape[:2]), max(image.shape[:2])), image)

        return {
            "torched_image": tensor_from_rgb_image(pad_dict["image"]),
            "image_path": str(image_path),
            "pads": pad_dict["pads"],
            "original_width": width,
            "original_height": height,
        }


def main():
    args = get_args()
    torch.distributed.init_process_group(backend="nccl")

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    hparams.update(
        {
            "local_rank": args.local_rank,
            "fp16": args.fp16,
        }
    )

    output_mask_path = args.output_path
    output_mask_path.mkdir(parents=True, exist_ok=True)
    hparams["output_mask_path"] = output_mask_path

    device = torch.device("cuda", args.local_rank)

    model = object_from_dict(hparams["model"])
    model = model.to(device)

    if args.fp16:
        model = model.half()

    corrections: Dict[str, str] = {"model.": ""}
    state_dict = state_dict_from_disk(file_path=args.weight_path, rename_in_layers=corrections)
    model.load_state_dict(state_dict)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    file_paths = []

    for regexp in ["*.jpg", "*.png", "*.jpeg", "*.JPG"]:
        file_paths += sorted([x for x in tqdm(args.input_path.rglob(regexp))])

    # Filter file paths for which we already have predictions
    file_paths = [x for x in file_paths if not (args.output_path / x.parent.name / f"{x.stem}.png").exists()]

    dataset = InferenceDataset(file_paths, transform=from_dict(hparams["test_aug"]))

    sampler = DistributedSampler(dataset, shuffle=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        sampler=sampler,
    )

    predict(dataloader, model, hparams, device)


def predict(dataloader, model, hparams, device):
    model.eval()

    if hparams["local_rank"] == 0:
        loader = tqdm(dataloader)
    else:
        loader = dataloader

    with torch.no_grad():
        for batch in loader:
            torched_images = batch["torched_image"]  # images that are rescaled and padded

            if hparams["fp16"]:
                torched_images = torched_images.half()

            image_paths = batch["image_path"]
            pads = batch["pads"]
            heights = batch["original_height"]
            widths = batch["original_width"]

            batch_size = torched_images.shape[0]

            predictions = model(torched_images.to(device))

            for batch_id in range(batch_size):
                file_id = Path(image_paths[batch_id]).stem
                folder_name = Path(image_paths[batch_id]).parent.name

                mask = (predictions[batch_id][0].cpu().numpy() > 0).astype(np.uint8) * 255
                mask = unpad_from_size(pads, image=mask)["image"]
                mask = cv2.resize(
                    mask, (widths[batch_id].item(), heights[batch_id].item()), interpolation=cv2.INTER_NEAREST
                )

                (hparams["output_mask_path"] / folder_name).mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(hparams["output_mask_path"] / folder_name / f"{file_id}.png"), mask)


if __name__ == "__main__":
    main()
