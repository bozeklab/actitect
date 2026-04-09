from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import yaml

import logging

logger = logging.getLogger(__name__)

__all__ = ['LabelMapping', 'get_label_mappings']


@dataclass
class LabelMapping:
    mappings: Dict[str, Dict[str, List[str]]]

    def get(self, name: str) -> Dict[str, List[str]]:
        if name not in self.mappings:
            available = ", ".join(sorted(self.mappings.keys()))
            raise KeyError(f"Label mapping '{name}' not found. Available mappings: {available}")
        return self.mappings[name]

def get_label_mappings(yaml_path: Path = None) -> LabelMapping:
    yaml_path = Path(yaml_path) if yaml_path is not None else Path(__file__).parent / 'label_mappings.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid label mapping YAML structure in {yaml_path}")

    return LabelMapping(mappings=data)

