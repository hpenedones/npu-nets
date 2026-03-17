"""Shared geometry helpers for the curated conveyor-belt path."""

ROWS_PER_COL = 4


def snake_tile_order(num_cols: int):
    """Return the serpentine compute-tile traversal for ``num_cols`` columns."""
    if not 1 <= num_cols <= 8:
        raise ValueError(f"num_cols must be in [1, 8], got {num_cols}")

    tiles = []
    for col in range(num_cols):
        rows = range(2, 6) if col % 2 == 0 else range(5, 1, -1)
        tiles.extend((col, row) for row in rows)
    return tiles
