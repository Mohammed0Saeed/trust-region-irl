import hydrax
from pathlib import Path


# get paths
def get_root_path():
    path = Path(hydrax.__path__[0]).resolve()
    return path
