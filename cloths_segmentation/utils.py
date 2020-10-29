from pathlib import Path
from typing import Union, Dict, List, Tuple


def get_id2_file_paths(path: Union[str, Path]) -> Dict[str, Path]:
    return {x.stem: x for x in Path(path).glob("*.*")}


def get_samples(image_path: Path, mask_path: Path) -> List[Tuple[Path, Path]]:
    """Couple masks and images.

    Args:
        image_path:
        mask_path:

    Returns:
    """

    image2path = get_id2_file_paths(image_path)
    mask2path = get_id2_file_paths(mask_path)

    return [(image_file_path, mask2path[file_id]) for file_id, image_file_path in image2path.items()]
