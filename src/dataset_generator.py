from src.util import DTOEncoder
from src.extraction import *
import numpy as np


def extract_prompt_from(data_path: str, task_name: str):
    X = parse_dataset_to_prompt(data_path, task_name)
    data_file_name = data_path.split("/")[-1]
    data_file_name = data_file_name.split("_")
    purpose = data_file_name[2]
    prompt_path = os.path.join(
        "./data/"
        f"{task_name}_{purpose}_X.json",
    )
    os.makedirs("./data/", exist_ok=True)

    with open(prompt_path, "w", encoding="utf-8") as output:
        json.dump(X, output, cls=DTOEncoder, indent=4)


def extract_labels_from(data_path, task_name: str):
    with open(data_path, "r", encoding="utf-8") as file:
        outputs = json.load(file)
        outputs = outputs["golds"]
        data_file_name = data_path.split("/")[-1]
        data_file_name = data_file_name.split("_")
        purpose = data_file_name[2]
        labels_path = os.path.join(
            "./data/"
            f"{task_name}_{purpose}_y.json",
        )
        os.makedirs("./data/", exist_ok=True)

    with open(labels_path, "w", encoding="utf-8") as output:
        json.dump(outputs, output, cls=DTOEncoder, indent=4)


def generate_np_X_y(purpose: str, task: str):
    X_data_path = f"./data/{task}_{purpose}_X.json"
    y_data_path = f"./data/{task}_{purpose}_y.json"
    X_data = []
    y_data = []
    X_outputs = None
    y_outputs = None
    with open(X_data_path, "r", encoding="utf-8") as X_file:
        X_outputs = json.load(X_file)
    with open(y_data_path, "r", encoding="utf-8") as y_file:
        y_outputs = json.load(y_file)
    for features in y_outputs:
        label_id = features["id"]
        label = features["output"]
        x = X_outputs[label_id]
        x = '\n'.join(x)
        X_data.append(x)
        y_data.append(label)
    return np.array(X_data), np.array(y_data)


def generate_np_test_X(task):
    X_test_path = f"./data/{task}_test_X.json"
    X_test = []
    with open(X_test_path, "r", encoding="utf-8") as X_file:
        X_outputs = json.load(X_file)
        for x_test in X_outputs.values():
            x = '\n'.join(x_test)
            X_test.append(x)
    return np.array(X_test)