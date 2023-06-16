import os
from pathlib import Path
from typing import Union

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".aif",
    ".aifc",
}


def list_files(
    path: Union[Path, str],
    extensions: set[str] = None,
    recursive: bool = False,
    sort: bool = True,
) -> list[Path]:
    """List files in a directory.

    Args:
        path (Path): Path to the directory.
        extensions (set, optional): Extensions to filter. Defaults to None.
        recursive (bool, optional): Whether to search recursively. Defaults to False.
        sort (bool, optional): Whether to sort the files. Defaults to True.

    Returns:
        list: List of files.
    """

    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist.")

    files = (
        [
            Path(os.path.join(root, filename))
            for root, _, filenames in os.walk(path, followlinks=True)
            for filename in filenames
            if Path(os.path.join(root, filename)).is_file()
        ]
        if recursive
        else [f for f in path.glob("*") if f.is_file()]
    )

    if extensions is not None:
        files = [f for f in files if f.suffix in extensions]

    if sort:
        files = sorted(files)

    return files
