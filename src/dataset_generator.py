from src.utils import (
    DTOEncoder,
    extract_prompt_tokens_stats,
    extract_output_tokens_stats,
)
from src.extraction import *


def extract_prompt_from(data_path: str, task_name: str):
    X = parse_dataset_to_prompt(data_path, task_name)
    data_file_name = data_path.split("/")[-1]
    data_file_name = data_file_name.split("_")
    purpose = data_file_name[2]
    prompt_path = os.path.join(
        "./data/" f"{task_name}_{purpose}_X.json",
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
            "./data/" f"{task_name}_{purpose}_y.json",
        )
        os.makedirs("./data/", exist_ok=True)

    with open(labels_path, "w", encoding="utf-8") as output:
        json.dump(outputs, output, cls=DTOEncoder, indent=4)


def generate_X_y_save(purpose: str, task: str):
    X_data_path = f"./data/{task}_{purpose}_X.json"
    y_data_path = f"./data/{task}_{purpose}_y.json"
    X_data_output_path = f"./data/{task}_{purpose}_X_only_str.json"
    y_data_output_path = f"./data/{task}_{purpose}_y_only_str.json"
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
        x = "\n".join(x)
        X_data.append(x)
        y_data.append(label)
    with open(X_data_output_path, "w", encoding="utf-8") as X_output:
        json.dump(X_data, X_output, cls=DTOEncoder, indent=4)
    with open(y_data_output_path, "w", encoding="utf-8") as y_output:
        json.dump(y_data, y_output, cls=DTOEncoder, indent=4)
    return X_data, y_data


def generate_test_X(task):
    X_test_path = f"./data/{task}_test_X.json"
    X_test = []
    with open(X_test_path, "r", encoding="utf-8") as X_file:
        X_outputs = json.load(X_file)
        for x_test in X_outputs.values():
            x = "\n".join(x_test)
            X_test.append(x)
    return X_test


def generate_X_y(purpose: str, task: str):
    X_data_output_path = f"./data/{task}_{purpose}_X_only_str.json"
    y_data_output_path = f"./data/{task}_{purpose}_y_only_str.json"
    X_data = []
    y_data = []
    X_outputs = None
    y_outputs = None
    with open(X_data_output_path, "r", encoding="utf-8") as X_file:
        X_outputs = json.load(X_file)
        for data in X_outputs:
            X_data.append(data)
    with open(y_data_output_path, "r", encoding="utf-8") as y_file:
        y_outputs = json.load(y_file)
        for data in y_outputs:
            y_data.append(data)
    return X_data, y_data


def generate_limited_questions_outpus(
    question_path: str,
    outputs_path: str,
    limited: int,
    limited_question_path: str,
    limited_outputs_path: str,
):
    questions = None
    outputs = None
    with open(question_path, "r", encoding="utf-8") as question_file:
        questions = json.load(question_file)
    with open(outputs_path, "r", encoding="utf-8") as outputs_file:
        outputs = json.load(outputs_file)
    new_questions = questions[:limited]
    new_outputs = {"task": outputs["task"], "golds": outputs["golds"][:limited]}
    with open(limited_question_path, "w", encoding="utf-8") as limited_question_file:
        json.dump(new_questions, limited_question_file, cls=DTOEncoder, indent=4)
    with open(limited_outputs_path, "w", encoding="utf-8") as limited_outputs_file:
        json.dump(new_outputs, limited_outputs_file, cls=DTOEncoder, indent=4)


def verify_output_length(outputs_path: str, limited: int):
    with open(outputs_path, "r", encoding="utf-8") as outputs_file:
        outputs = json.load(outputs_file)
        return len(outputs["golds"]) == limited


def verify_questions_length(questions_path: str, limited: int):
    with open(questions_path, "r", encoding="utf-8") as questions_file:
        questions = json.load(questions_file)
        return len(questions) == limited


limited = 100
old_questions_path = "./data/orginal_data/LaMP_1_dev_questions.json"
old_outputs_path = "./data/orginal_data/LaMP_1_dev_outputs.json"
new_questions_path = f"./data/LaMP_11_dev_questions.json"
new_outputs_path = f"./data/LaMP_11_dev_outputs.json"

generate_limited_questions_outpus(
    question_path=old_questions_path,
    outputs_path=old_outputs_path,
    limited=limited,
    limited_question_path=new_questions_path,
    limited_outputs_path=new_outputs_path,
)

print(verify_output_length(new_outputs_path, limited))
print(verify_questions_length(new_questions_path, limited))
