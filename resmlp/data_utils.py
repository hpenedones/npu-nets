"""HIGGS-only dataset utilities for the curated residual-MLP workflows."""

import gzip
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

DEFAULT_VAL_SIZE = 100_000
DEFAULT_SPLIT_SEED = 1234
HIGGS_INPUT_DIM = 28
HIGGS_TEST_FRACTION = 0.1
HIGGS_DATASET_CONFIG = {
    'kind': 'tabular',
    'input_dim': HIGGS_INPUT_DIM,
    'num_classes': 2,
    'test_fraction': HIGGS_TEST_FRACTION,
}
SUPPORTED_DATASETS = ('higgs',)
SUPPORTED_TRAIN_AUGS = ('none',)


class NormalizedTabularDataset(Dataset):
    def __init__(self, features, labels, *, mean, std):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)
        if self.features.ndim != 2:
            raise ValueError(f'Expected 2D features, got shape {tuple(self.features.shape)}')
        if self.labels.ndim != 1 or self.labels.shape[0] != self.features.shape[0]:
            raise ValueError('Labels must be a 1D tensor aligned with features')

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (self.features[idx] - self.mean) / self.std, int(self.labels[idx])


def _find_higgs_path(data_dir, names):
    data_dir = Path(data_dir)
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    return None


def _prepare_higgs_tensors(features, labels):
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if features.ndim != 2:
        raise ValueError(f'HIGGS features must be 2D, got shape {tuple(features.shape)}')
    if features.shape[1] != HIGGS_INPUT_DIM:
        raise ValueError(
            f'HIGGS features must have {HIGGS_INPUT_DIM} columns, got {features.shape[1]}'
        )
    labels = labels.view(-1)
    if labels.shape[0] != features.shape[0]:
        raise ValueError('HIGGS labels/features row count mismatch')
    return features, labels


def _load_higgs_cache(data_dir='data'):
    data_dir = Path(data_dir)
    cache_path = _find_higgs_path(data_dir, ('HIGGS.pt', 'higgs.pt'))
    if cache_path is not None:
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        if not isinstance(data, dict):
            raise ValueError(f'Expected HIGGS tensor cache dict at {cache_path}')
        if {'train_features', 'train_labels', 'test_features', 'test_labels'} <= set(data):
            train_features, train_labels = _prepare_higgs_tensors(
                data['train_features'], data['train_labels']
            )
            test_features, test_labels = _prepare_higgs_tensors(
                data['test_features'], data['test_labels']
            )
            return {
                'train_features': train_features,
                'train_labels': train_labels,
                'test_features': test_features,
                'test_labels': test_labels,
            }
        if 'features' in data and 'labels' in data:
            features, labels = _prepare_higgs_tensors(data['features'], data['labels'])
            return {'features': features, 'labels': labels}
        raise ValueError(
            'Expected HIGGS tensor cache dict with either features/labels or '
            f'train_features/train_labels/test_features/test_labels at {cache_path}'
        )

    raw_path = _find_higgs_path(
        data_dir,
        ('HIGGS.csv.gz', 'HIGGS.csv', 'higgs.csv.gz', 'higgs.csv'),
    )
    if raw_path is None:
        raise FileNotFoundError(
            f'HIGGS dataset not found under {data_dir}. Expected one of: '
            'HIGGS.pt, higgs.pt, HIGGS.csv.gz, HIGGS.csv, higgs.csv.gz, higgs.csv'
        )

    opener = gzip.open if raw_path.suffix == '.gz' else open
    with opener(raw_path, 'rt') as handle:
        table = np.loadtxt(handle, delimiter=',', dtype=np.float32)
    if table.ndim != 2 or table.shape[1] < 1 + HIGGS_INPUT_DIM:
        raise ValueError(
            f'HIGGS raw file must have at least {1 + HIGGS_INPUT_DIM} columns, got shape {table.shape}'
        )

    labels = table[:, 0].astype(np.int64, copy=False)
    features = table[:, 1 : 1 + HIGGS_INPUT_DIM]
    features_t, labels_t = _prepare_higgs_tensors(features, labels)
    torch.save({'features': features_t, 'labels': labels_t}, data_dir / 'HIGGS.pt')
    return {'features': features_t, 'labels': labels_t}


def load_higgs_datasets(data_dir='data', *, split_seed=DEFAULT_SPLIT_SEED):
    cache = _load_higgs_cache(data_dir=data_dir)
    if {'train_features', 'train_labels', 'test_features', 'test_labels'} <= set(cache):
        train_features = cache['train_features']
        train_labels = cache['train_labels']
        test_features = cache['test_features']
        test_labels = cache['test_labels']
        if train_features.shape[0] < 2:
            raise ValueError('HIGGS training split must contain at least 2 rows')
        if test_features.shape[0] < 1:
            raise ValueError('HIGGS test split must contain at least 1 row')

        mean = train_features.mean(dim=0)
        std = train_features.std(dim=0).clamp_min(1e-6)
        train_ds = NormalizedTabularDataset(train_features, train_labels, mean=mean, std=std)
        test_ds = NormalizedTabularDataset(test_features, test_labels, mean=mean, std=std)
        return train_ds, test_ds

    features = cache['features']
    labels = cache['labels']
    total = features.shape[0]
    if total < 2:
        raise ValueError('HIGGS dataset must contain at least 2 rows')

    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-6)
    full_ds = NormalizedTabularDataset(features, labels, mean=mean, std=std)

    test_size = max(1, int(round(total * HIGGS_TEST_FRACTION)))
    if test_size >= total:
        test_size = total // 10 or 1
    generator = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(total, generator=generator)
    test_idx = perm[:test_size].tolist()
    train_idx = perm[test_size:].tolist()
    return Subset(full_ds, train_idx), Subset(full_ds, test_idx)


def normalize_dataset_name(dataset_name):
    key = 'higgs' if dataset_name is None else dataset_name.lower()
    if key != 'higgs':
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected: higgs")
    return key


def resolve_dataset_name(requested_dataset=None, checkpoint_dataset=None):
    requested = normalize_dataset_name(requested_dataset) if requested_dataset else None
    checkpoint = normalize_dataset_name(checkpoint_dataset) if checkpoint_dataset else None
    if requested and checkpoint and requested != checkpoint:
        raise ValueError(
            f"Checkpoint was trained for dataset '{checkpoint}', but '{requested}' was requested"
        )
    return requested or checkpoint or 'higgs'


def get_dataset_config(dataset_name):
    normalize_dataset_name(dataset_name)
    return dict(HIGGS_DATASET_CONFIG)


def normalize_train_aug(train_aug):
    key = train_aug.lower()
    if key != 'none':
        raise ValueError("Unsupported train augmentation. The curated main branch supports only 'none'.")
    return key


def load_datasets(dataset_name='higgs', data_dir='data', *, train_aug='none', split_seed=DEFAULT_SPLIT_SEED):
    normalize_dataset_name(dataset_name)
    normalize_train_aug(train_aug)
    return load_higgs_datasets(data_dir=data_dir, split_seed=split_seed)


def split_train_val(train_ds, val_size=DEFAULT_VAL_SIZE, split_seed=DEFAULT_SPLIT_SEED):
    if not 0 <= val_size < len(train_ds):
        raise ValueError(f'val_size must be in [0, {len(train_ds) - 1}], got {val_size}')
    if val_size == 0:
        return train_ds, None

    train_size = len(train_ds) - val_size
    generator = torch.Generator().manual_seed(split_seed)
    return random_split(train_ds, [train_size, val_size], generator=generator)


def get_dataset_dataloaders(
    dataset_name,
    batch_size,
    *,
    data_dir='data',
    train_aug='none',
    val_size=DEFAULT_VAL_SIZE,
    split_seed=DEFAULT_SPLIT_SEED,
    train_num_workers=2,
    eval_num_workers=None,
    pin_memory=True,
    drop_last_train=False,
    eval_batch_size=None,
):
    train_full, test_ds = load_datasets(
        dataset_name,
        data_dir=data_dir,
        train_aug=train_aug,
        split_seed=split_seed,
    )
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
    data_dir='data',
    val_size=DEFAULT_VAL_SIZE,
    split_seed=DEFAULT_SPLIT_SEED,
):
    train_ds, test_ds = load_datasets(dataset_name, data_dir=data_dir, split_seed=split_seed)
    _, val_ds = split_train_val(train_ds, val_size=val_size, split_seed=split_seed)

    if split == 'val':
        if val_ds is None:
            raise ValueError('Validation split requested, but val_size=0')
        return val_ds
    if split == 'test':
        return test_ds
    raise ValueError(f'Unknown split: {split}')
