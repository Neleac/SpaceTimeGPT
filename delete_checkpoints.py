import os
import shutil

for root, dirs, files in os.walk("training"):
    for folder in dirs:
        if folder[:len("checkpoint")] == "checkpoint":
            shutil.rmtree(os.path.join(root, folder))
