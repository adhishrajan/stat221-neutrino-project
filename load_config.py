"""
Load configuration from config.yaml
"""

import yaml
import os

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_signal_params(config=None):
    """Get signal (astrophysical) parameters from config"""
    if config is None:
        config = load_config()
    return config['signal']

def get_background_params(config=None):
    """Get background (atmospheric) parameters from config"""
    if config is None:
        config = load_config()
    return config['background']

def get_energy_bins_params(config=None):
    """Get energy binning parameters from config"""
    if config is None:
        config = load_config()
    return config['energy_bins']

def get_sampler_params(sampler_name, config=None):
    """Get sampler settings from config"""
    if config is None:
        config = load_config()
    return config['samplers'][sampler_name]

def get_hierarchical_params(config=None):
    """Get hierarchical model parameters from config"""
    if config is None:
        config = load_config()
    return config['hierarchical']

if __name__ == "__main__":
    config = load_config()
    print("Loaded config:")
    print(config)
