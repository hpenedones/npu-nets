from hashlib import sha1
from pathlib import Path


def source_fingerprint(*paths: Path) -> str:
    digest = sha1()
    for path in paths:
        resolved = Path(path).resolve()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(resolved.read_bytes())
    return digest.hexdigest()[:10]


def sgd_lr_token(sgd_lr: float) -> str:
    value = format(sgd_lr, ".8g")
    return f"lr{value.replace('-', 'm').replace('+', '').replace('.', 'p')}"


def training_kernel_tag(project_dir: Path, *, B: int, H: int, sgd_lr: float) -> str:
    kernels_dir = Path(project_dir) / "aie_kernels"
    return (
        f"b{B}_h{H}_{sgd_lr_token(sgd_lr)}_"
        f"{source_fingerprint(kernels_dir / 'matmul_relu_skip.cc', kernels_dir / 'residual_backward.cc', kernels_dir / 'copy_activation.cc')}"
    )


def training_kernel_archive_name(project_dir: Path, *, B: int, H: int, sgd_lr: float) -> str:
    return f"resmlp_training_kernels_{training_kernel_tag(project_dir, B=B, H=H, sgd_lr=sgd_lr)}.a"


def full_training_kernel_tag(
    project_dir: Path,
    *,
    B: int,
    H: int,
    embed_chunk_rows: int,
    n_cls_padded: int,
    sgd_lr: float,
) -> str:
    kernels_dir = Path(project_dir) / "aie_kernels"
    return (
        f"b{B}_h{H}_ec{embed_chunk_rows}_cls{n_cls_padded}_{sgd_lr_token(sgd_lr)}_"
        f"{source_fingerprint(kernels_dir / 'matmul_relu_skip.cc', kernels_dir / 'residual_backward.cc', kernels_dir / 'copy_activation.cc', kernels_dir / 'embed_forward.cc', kernels_dir / 'embed_backward.cc', kernels_dir / 'head_forward_loss.cc', kernels_dir / 'head_backward.cc')}"
    )


def full_training_kernel_archive_name(
    project_dir: Path,
    *,
    B: int,
    H: int,
    embed_chunk_rows: int,
    n_cls_padded: int,
    sgd_lr: float,
) -> str:
    return (
        "full_training_kernels_"
        f"{full_training_kernel_tag(project_dir, B=B, H=H, embed_chunk_rows=embed_chunk_rows, n_cls_padded=n_cls_padded, sgd_lr=sgd_lr)}.a"
    )
