from pathlib import Path

import yaml


def load_yaml_config(path):
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config at {config_path} must be a YAML mapping.")
    return data


def load_config_with_base(path, section_name):
    config_path = Path(path)
    root = load_yaml_config(config_path)
    if "train_config" in root:
        train_config_path = resolve_path(config_path, root["train_config"])
        base = load_yaml_config(train_config_path)
    else:
        base = root

    section = root.get(section_name, {})
    if section and not isinstance(section, dict):
        raise TypeError(f"Section '{section_name}' in {config_path} must be a YAML mapping.")

    merged = dict(base)
    merged.update(section)
    return config_path, root, merged


def resolve_path(config_path, value):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config_path).parent / path
