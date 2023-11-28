from src.util import DTOEncoder
from src.extraction import *


def extract_prompt_from(data_path: str, task_name: str):
    X = parse_dataset_to_prompt(data_path, task_name)
    data_file_name = data_path.split("/")[-1]
    data_file_name = data_file_name.split("_")
    purpose = data_file_name[3]
    prompt_path = os.path.join(
        "./data/"
        f"{task_name}_{purpose}_X.json",
    )
    os.makedirs("./data/", exist_ok=True)

    with open(prompt_path, "w", encoding="utf-8") as output:
        json.dump(X, output, cls=DTOEncoder, indent=4)


paths = ["./data/LaMP_1_train_questions.json", "./data/LaMP_1_dev_questions.json", "./data/LaMP_1_test_questions.json"]
for path in paths:
    extract_prompt_from(path, "LaMP_1")