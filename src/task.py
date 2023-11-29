import json
import os
from typing import Any, Callable, Dict, List

from src.util import DTOEncoder

from src.util import labels
from src.util import default_data_path, default_extract_path, default_prompt_path
from src.extraction import parse_dataset_to_prompt
from src.evaluation import LaMPEvaluation


class LaMPTask:
    store_path: str  # where query, label, and results are stored
    extract_path: str  # where the results/lables are extract to for evaluation
    prompt_save_path: str
    
    subscribers: Dict[str, Callable[[Dict[str, str], labels], None]]

    task_question_file: str
    task_output_file: str
    task_id: str
    task_type: str
    task_name: str  # name of LaMP dataset (LaMP_1 or LaMP_2 in this case)
    
    parsed_prompts: Dict[str, str]
    evaluation: LaMPEvaluation
    preds: Dict[str, labels]
    score: List[Dict]

    def __init__(
        self,
        task_question_file: str,
        task_output_file: str,
        subscribers: Dict[str, Callable[[Dict[str, str], labels], None]],
        prompt_save_path: str = None,
        save_lables: bool = True,
        store_path: str = default_data_path,
        extract_path: str = default_extract_path,
    ) -> None:

        assert (subscribers is not None and isinstance(subscribers, dict) and len(subscribers) > 0), "Need feeds function to feed prompt into language model"

        assert (isinstance(list(subscribers.values())[0], Callable)), "subscriber(s) need to be a function"

        self.subscribers = subscribers

        self.store_path = ""
        if save_lables:
            self.store_path = store_path

        self.extract_path = extract_path
        self.prompt_save_path=prompt_save_path

        self.parsed_prompts = None
        self.preds = None
        self.score = None

        _, q_id, q_type, *_ = task_question_file.split("_")
        _, o_id, o_type, *_ = task_output_file.split("_")
        assert (
            q_id == o_id and q_type == o_type
        ), "question file and output file id not match"
        self.task_id = q_id
        self.task_type = q_type

        self.task_question_file = task_question_file
        self.task_output_file = task_output_file

        with open(task_output_file, "r", encoding="utf-8") as file:
            contents = json.load(file)
            self.task_name = contents["task"]

        self.evaluation = LaMPEvaluation(
            single_gold_json_file_addr=self.task_output_file
        )

    def run(self) -> Any:
        self.parsed_prompts = parse_dataset_to_prompt(
            self.task_question_file, self.task_name
        )

        if self.prompt_save_path is not None and self.prompt_save_path != "":
            prompt_save_name = os.path.join(
                    self.prompt_save_path,
                    f"LaMP_{self.task_id}_{self.task_type}_prompts.json",
                )
            with open(prompt_save_name, "w", encoding="utf-8") as output:
                json.dump(self.parsed_prompts, output, cls=DTOEncoder, indent=4)
        
        self.preds = dict()
        self.score = dict()
        # feed prompt into subscriber
        for subscriber_name, subscriber_func in self.subscribers.items():
            curr_preds = labels(task=self.task_name, golds=list())
            subscriber_func(self.parsed_prompts, curr_preds)
            
            self.preds[subscriber_name] = curr_preds

            if self.store_path is not None and self.store_path != "":
                preds_save_name = os.path.join(
                    self.store_path,
                    f"LaMP_{self.task_id}_{self.task_type}_preds_{subscriber_name}.json",
                )
                os.makedirs(self.store_path, exist_ok=True)
                with open(preds_save_name, "w", encoding="utf-8") as output:
                    json.dump(curr_preds, output, cls=DTOEncoder, indent=4)

            self.score[subscriber_name] = self.evaluation.evaluate_task(preds_save_name, self.task_name)
        return self.score
