import os
import torch
import numpy as np
import subprocess
import shutil
import random


def seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def get_file_list():
    return [
        b.decode()
        for b in set(
            subprocess.check_output(
                'git ls-files -- ":!:load/*"', shell=True
            ).splitlines()
        )
        | set(  # hard code, TODO: use config to exclude folders or files
            subprocess.check_output(
                "git ls-files --others --exclude-standard", shell=True
            ).splitlines()
        )
    ]

def save_code_snapshot(model_path):
    print("savesavesave")
    os.makedirs(model_path, exist_ok=True)

    for f in get_file_list():
        if not (f.startswith('models/') or f.startswith('data/')):
            continue
        if not os.path.exists(f) or os.path.isdir(f):
            continue
        os.makedirs(os.path.join(model_path, os.path.dirname(f)), exist_ok=True)
        shutil.copyfile(f, os.path.join(model_path, f))
