import yaml

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