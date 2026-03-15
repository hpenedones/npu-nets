"""Backward-compatible MNIST wrappers around the shared dataset helpers."""

from resmlp.data_utils import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_SIZE,
    dataset_transform,
    denormalize_images,
    get_dataset_config,
    get_dataset_dataloaders,
    get_eval_dataset,
    load_datasets,
    save_prediction_preview as save_dataset_prediction_preview,
    split_train_val,
)

MNIST_MEAN = get_dataset_config("mnist")["mean"]
MNIST_STD = get_dataset_config("mnist")["std"]


def mnist_transform():
    return dataset_transform("mnist")


def load_mnist_datasets(data_dir="data"):
    return load_datasets("mnist", data_dir=data_dir)


def split_mnist_train_val(train_ds, val_size=DEFAULT_VAL_SIZE, split_seed=DEFAULT_SPLIT_SEED):
    return split_train_val(train_ds, val_size=val_size, split_seed=split_seed)


def get_mnist_dataloaders(batch_size, **kwargs):
    return get_dataset_dataloaders("mnist", batch_size, **kwargs)


def get_mnist_eval_dataset(*, split, **kwargs):
    return get_eval_dataset("mnist", split=split, **kwargs)


def denormalize_mnist(images):
    return denormalize_images(images, "mnist")


def save_prediction_preview(images, labels, preds, out_path, **kwargs):
    return save_dataset_prediction_preview(
        images,
        labels,
        preds,
        out_path,
        dataset_name="mnist",
        **kwargs,
    )
