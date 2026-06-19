import os
import logging
import pandas as pd

def get_workspace_dir():
    # Workspace root directory
    return r"d:\Sarthak\Hackathons\gridlock hackathon\set2\antitry"

def get_data_path():
    return os.path.join(get_workspace_dir(), "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")

def setup_logger(name="traffic_ml"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler inside reports directory
        os.makedirs(os.path.join(get_workspace_dir(), "reports"), exist_ok=True)
        log_file = os.path.join(get_workspace_dir(), "reports", f"{name}.log")
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def load_data():
    path = get_data_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)
