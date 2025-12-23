import yaml
import hashlib
# ------------------------------------------------------------
# ---------------------Utility Functions----------------------
# ------------------------------------------------------------


def compute_paper_id(source: str, source_id: str) -> str:
    """
    Computes Unique SHA-256 Hash for each file, serving as a paper_id

    Args:
        source (str): Source publication of the paper
        source_id (str): Entry id of paper according to publication website

    returns:
        str: A hash for according to content
    """
    raw = f"{source} :: {source_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
    
def load_yaml(path: str) -> dict:
    """
    Load the yaml file present at path into a dictionary

    Args:
        path (str): yaml file source path

    returns:
        dict: contents of yaml file as a dictionary
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)