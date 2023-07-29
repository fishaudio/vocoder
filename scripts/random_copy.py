import random
from pathlib import Path

import click
from loguru import logger


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("dst", type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument("num", type=int)
@click.option("--seed", help="Random seed", type=int, default=42)
def random_copy(src: Path, dst: Path, num: int, seed: int):
    """Copy random files from SRC to DST."""

    src, dst = Path(src), Path(dst)

    files = [f for f in src.rglob("*") if f.is_file() and f.suffix in [".wav", ".flac"]]
    logger.info(f"Found {len(files)} files in {src}")

    generator = random.Random(seed)
    selected_files = generator.sample(files, num)

    logger.info(f"Copying {len(selected_files)} files to {dst}")

    for f in selected_files:
        f_dst = dst / f.relative_to(src)
        f_dst.parent.mkdir(parents=True, exist_ok=True)
        f_dst.write_bytes(f.read_bytes())

    logger.info("Done")


if __name__ == "__main__":
    random_copy()
