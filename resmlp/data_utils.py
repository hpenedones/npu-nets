"""Shared dataset and preview helpers for ResMLP workflows."""

from math import ceil
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DEFAULT_VAL_SIZE = 10_000
DEFAULT_SPLIT_SEED = 1234

DATASET_CONFIGS = {
    "mnist": {
        "input_dim": 28 * 28,
        "num_classes": 10,
        "mean": (0.1307,),
        "std": (0.3081,),
        "image_size": (28, 28),
        "channels": 1,
    },
    "cifar10": {
        "input_dim": 3 * 32 * 32,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "image_size": (32, 32),
        "channels": 3,
    },
}
SUPPORTED_DATASETS = tuple(DATASET_CONFIGS)

_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


def normalize_dataset_name(dataset_name):
    key = dataset_name.lower()
    if key not in DATASET_CONFIGS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: {supported}")
    return key


def resolve_dataset_name(requested_dataset=None, checkpoint_dataset=None):
    requested = normalize_dataset_name(requested_dataset) if requested_dataset else None
    checkpoint = normalize_dataset_name(checkpoint_dataset) if checkpoint_dataset else None
    if requested and checkpoint and requested != checkpoint:
        raise ValueError(
            f"Checkpoint was trained for dataset '{checkpoint}', but '{requested}' was requested"
        )
    return requested or checkpoint or "mnist"


def get_dataset_config(dataset_name):
    return dict(DATASET_CONFIGS[normalize_dataset_name(dataset_name)])


def dataset_transform(dataset_name):
    config = get_dataset_config(dataset_name)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ]
    )


def load_datasets(dataset_name, data_dir="data"):
    dataset_name = normalize_dataset_name(dataset_name)
    transform = dataset_transform(dataset_name)
    if dataset_name == "mnist":
        dataset_cls = datasets.MNIST
    elif dataset_name == "cifar10":
        dataset_cls = datasets.CIFAR10
    else:
        raise AssertionError(f"Unhandled dataset: {dataset_name}")

    train_ds = dataset_cls(data_dir, train=True, download=True, transform=transform)
    test_ds = dataset_cls(data_dir, train=False, download=True, transform=transform)
    return train_ds, test_ds


def split_train_val(train_ds, val_size=DEFAULT_VAL_SIZE, split_seed=DEFAULT_SPLIT_SEED):
    if not 0 <= val_size < len(train_ds):
        raise ValueError(f"val_size must be in [0, {len(train_ds) - 1}], got {val_size}")

    if val_size == 0:
        return train_ds, None

    train_size = len(train_ds) - val_size
    generator = torch.Generator().manual_seed(split_seed)
    return random_split(train_ds, [train_size, val_size], generator=generator)


def get_dataset_dataloaders(
    dataset_name,
    batch_size,
    *,
    data_dir="data",
    val_size=DEFAULT_VAL_SIZE,
    split_seed=DEFAULT_SPLIT_SEED,
    train_num_workers=2,
    eval_num_workers=None,
    pin_memory=True,
    drop_last_train=False,
    eval_batch_size=None,
):
    train_full, test_ds = load_datasets(dataset_name, data_dir=data_dir)
    train_ds, val_ds = split_train_val(
        train_full,
        val_size=val_size,
        split_seed=split_seed,
    )

    if eval_num_workers is None:
        eval_num_workers = train_num_workers
    if eval_batch_size is None:
        eval_batch_size = batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
            pin_memory=pin_memory,
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def get_eval_dataset(
    dataset_name,
    *,
    split,
    data_dir="data",
    val_size=DEFAULT_VAL_SIZE,
    split_seed=DEFAULT_SPLIT_SEED,
):
    train_ds, test_ds = load_datasets(dataset_name, data_dir=data_dir)
    _, val_ds = split_train_val(
        train_ds,
        val_size=val_size,
        split_seed=split_seed,
    )

    if split == "val":
        if val_ds is None:
            raise ValueError("Validation split requested, but val_size=0")
        return val_ds
    if split == "test":
        return test_ds
    raise ValueError(f"Unknown split: {split}")


def denormalize_images(images, dataset_name):
    config = get_dataset_config(dataset_name)
    images = torch.as_tensor(images).detach().cpu().float()
    mean = torch.tensor(config["mean"], dtype=images.dtype).view(1, -1, 1, 1)
    std = torch.tensor(config["std"], dtype=images.dtype).view(1, -1, 1, 1)
    return images * std + mean


def save_prediction_preview(
    images,
    labels,
    preds,
    out_path,
    *,
    dataset_name,
    max_items=16,
    cols=4,
    scale=4,
):
    config = get_dataset_config(dataset_name)
    images = denormalize_images(images, dataset_name)
    labels = [int(x) for x in torch.as_tensor(labels).view(-1).tolist()]
    preds = [int(x) for x in torch.as_tensor(preds).view(-1).tolist()]
    count = min(len(labels), len(preds), len(images), max_items)
    if count <= 0:
        raise ValueError("No prediction samples provided for preview")

    cols = max(1, min(cols, count))
    rows = ceil(count / cols)
    image_h, image_w = config["image_size"]
    thumb_w = image_w * scale
    thumb_h = image_h * scale
    label_h = 18

    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for idx in range(count):
        row = idx // cols
        col = idx % cols
        x0 = col * thumb_w
        y0 = row * (thumb_h + label_h)

        img = images[idx].clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).numpy()
        if config["channels"] == 1:
            tile = Image.fromarray(img[:, :, 0], mode="L")
            tile = tile.resize((thumb_w, thumb_h), resample=_NEAREST).convert("RGB")
        elif config["channels"] == 3:
            tile = Image.fromarray(img, mode="RGB").resize(
                (thumb_w, thumb_h), resample=_NEAREST
            )
        else:
            raise ValueError(f"Unsupported channel count: {config['channels']}")
        canvas.paste(tile, (x0, y0))

        ok = preds[idx] == labels[idx]
        color = (0, 128, 0) if ok else (180, 0, 0)
        draw.rectangle((x0, y0, x0 + thumb_w - 1, y0 + thumb_h - 1), outline=color, width=2)
        draw.text((x0 + 2, y0 + thumb_h + 2), f"t={labels[idx]} p={preds[idx]}", fill=color)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path
