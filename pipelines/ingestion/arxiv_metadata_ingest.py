import time
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List
import feedparser
import yaml

from utils.logging import setup_logger, log_event


#------------------------------------------------------------
# Utility Functions------------------------------------------
# -----------------------------------------------------------
def load_yaml(path: str) -> dict:
    '''
    Load the yaml file present at path into a dictionary
    
    Args:
        path (str): yaml file source path
    
    returns:
        dict: contents of yaml file as a dictionary'''
    with open(path, 'r') as f:
        return yaml.safe_load(f)
        
def compute_paper_id(source: str, source_id: str) -> str:
    '''
    '''
    raw = f"{source} :: {source_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
    
    