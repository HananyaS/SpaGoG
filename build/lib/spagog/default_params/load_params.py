import os
import json
from pkg_resources import resource_filename


def load_params(model: str):
    assert model in ["gc", "gnc", "gc+nc"]

    abs_path = resource_filename(__name__, ".")
    if model == "gc":
        file_path = "gc.json"

        # with open("gc.json", "r") as f:
        #     return json.load(f)

    elif model == "gnc":
        file_path = "gnc.json"

        # with open("gnc.json", "r") as f:
        #     return json.load(f)

    else:
        file_path = "gc+nc.json"
    # with open("gc+nc.json", "r") as f:
    #     file_path = "gc+nc.json"

        # return json.load(f)
    
    with open(os.path.join(abs_path, file_path), "r") as f:
        return json.load(f)
        
