import subprocess

def get_dataset_hash():
    return subprocess.check_output(
        ["dvc", "status", "-c"],
        stderr=subprocess.DEVNULL
    ).decode().strip()
    
# metrics["dataset_hash"] = get_dataset_hash()