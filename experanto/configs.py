from pathlib import Path
from omegaconf import OmegaConf, open_dict
from hydra import compose, initialize, initialize_config_dir

# get config relative to this file
script_dir = Path(__file__).parent
config_path = script_dir / '..' / 'configs' / 'default.yaml'
config_path = config_path.resolve()
cfg = OmegaConf.load(config_path)

DEFAULT_CONFIG = cfg
DEFAULT_DATASET_CONFIG = cfg.dataset
DEFAULT_MODALITY_CONFIG = cfg.dataset.modality_config
DEFAULT_DATALOADER_CONFIG = cfg.dataloader
