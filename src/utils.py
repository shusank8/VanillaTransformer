import torch
import os

def load_model_if(filepath):
    files = os.listdir(filepath)
    if len(files)==0:
        return None
    else:
        files.sort()
        print(f"{filepath}/{files[-1]}")
        return f"{filepath}/{files[-1]}"
    