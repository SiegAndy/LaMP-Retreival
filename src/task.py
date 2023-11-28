import json
import os
from typing import Any, List

from src.util import DTOEncoder

from src.util import labels
from src.util import default_data_addr, default_extract_addr
from src.extraction import parse_dataset_to_prompt
from src.model import feed_prompt_to_lm
from src.evaluation import LaMPEvaluation


class LaMPTask:
    store_addr: str  # where query, label, and results are stored
    extract_addr: str  # where the results/lables are extract to for evaluation
    task_question_file: str
    task_output_file: str
    task_filename_component: List[str]
    task_name: str  # name of LaMP dataset (LaMP_1 or LaMP_2 in this case)
    evaluation: LaMPEvaluation
    preds: labels
    score: Any

    def __init__(
        self,
        task_question_file: str,
        task_output_file: str,
        store_addr: str = default_data_addr,
        extract_addr: str = default_extract_addr,
    ) -> None:

        self.store_addr = store_addr
        self.extract_addr = extract_addr
        _, q_id, q_type, *_ = task_question_file.split("_")
        _, o_id, o_type, *_ = task_output_file.split("_")
        assert (
            q_id == o_id and q_type == o_type
        ), "question file and output file id not match"
        self.task_filename_component = [q_id, q_type]

        self.task_question_file = task_question_file
        self.task_output_file = task_output_file

        with open(task_output_file, "r", encoding="utf-8") as file:
            contents = json.load(file)
            self.task_name = contents["task"]
        self.preds = labels(task=self.task_name, golds=list())

        self.evaluation = LaMPEvaluation(
            single_gold_json_file_addr=self.task_output_file
        )

    def run(self) -> Any:
        parsed_prompts = parse_dataset_to_prompt(
            self.task_question_file, self.task_name
        )
        feed_prompt_to_lm(parsed_prompts, self.preds)
        preds_save_name = os.path.join(
            self.store_addr,
            f"LaMP_{self.task_filename_component[0]}_{self.task_filename_component[1]}_preds.json",
        )

        os.makedirs(self.store_addr, exist_ok=True)

        with open(preds_save_name, "w", encoding="utf-8") as output:
            json.dump(self.preds, output, cls=DTOEncoder, indent=4)
        self.score = self.evaluation.evaluate_task(preds_save_name, self.task_name)
        return self.score
