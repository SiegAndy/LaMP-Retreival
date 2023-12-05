from genericpath import isfile
import json
import os
from queue import Queue
from threading import Thread
import time
from typing import Any, Callable, Dict, List, Tuple

from src.utils import (
    default_data_path,
    default_extract_path,
    DTOEncoder,
    labels,
)
from src.extraction import parse_dataset_to_prompt
from src.evaluation import LaMPEvaluation
from rich.progress import Progress, TaskID

default_total = 100


def subscriber_proc_thread(
    prog: Progress,
    task_dict: Dict[str, Tuple[TaskID, Callable, labels]],
    pipeline: Queue,
):
    """
    task_dict: dict of tuple of (taskID and subscriber func and labels)
    read content from pipeline (tuple of subscriber name and (id, prompt) and total number)
    """
    while True:
        content = pipeline.get()
        if content == "finished":
            break
        sub_name, (p_id, prompt), total_num = content
        task, sub_func, curr_labels = task_dict[sub_name]
        curr_label = sub_func(p_id, prompt)
        curr_labels.golds.append(curr_label)
        prog.update(task, advance=100 / total_num)


def subscriber_task_thread(
    prog: Progress,
    task_info: Tuple[str, TaskID],
    parsed_prompts: Dict[str, List[str]],
    pipeline: Queue,
):
    """
    task_info: tuple of subscriber name and taskID for progress
    send content to pipeline (tuple of subscriber name and (id, prompt) and total number)
    """
    sub_name, task = task_info
    total_num = len(parsed_prompts)
    for id, prompt in parsed_prompts.items():
        pipeline.put((sub_name, (id, prompt), total_num))
        prog.update(task, advance=default_total / total_num)
    time.sleep(1)


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

    parsed_prompts: Dict[str, List[str]]
    evaluation: LaMPEvaluation
    preds: Dict[str, labels]
    score: List[Dict]
    keyword_extraction: bool  # determine whether using keyword_extraction

    def __init__(
        self,
        task_question_file: str,
        task_output_file: str,
        subscribers: Dict[str, Callable[[Dict[str, str], labels], None]],
        prompt_save_path: str = None,
        save_lables: bool = True,
        run_eval: bool = False,
        store_path: str = default_data_path,
        extract_path: str = default_extract_path,
        keyword_extraction: bool = True,
        worker_count: int = 1,
    ) -> None:
        assert (
            subscribers is not None
            and isinstance(subscribers, dict)
            and len(subscribers) > 0
        ), "Need feeds function to feed prompt into language model"

        assert isinstance(
            list(subscribers.values())[0], Callable
        ), "subscriber(s) need to be a function"

        assert worker_count >= 1, "Must have at least one worker"

        self.worker_count = worker_count
        self.subscribers = subscribers
        self.store_path = ""
        if save_lables:
            self.store_path = store_path

        self.extract_path = extract_path
        self.prompt_save_path = prompt_save_path

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
        self.keyword_extraction = keyword_extraction
        self.suffix = "with_keyword" if self.keyword_extraction else "without_keyword"
        with open(task_output_file, "r", encoding="utf-8") as file:
            contents = json.load(file)
            self.task_name = contents["task"]

        self.evaluation = None
        if run_eval:
            self.evaluation = LaMPEvaluation(
                single_gold_json_file_addr=self.task_output_file
            )

    def evaluate(self, preds_save_name: List[str] = None):
        save_name_not_provided = preds_save_name is None or len(preds_save_name) == 0
        assert save_name_not_provided or len(preds_save_name) == len(
            self.subscribers.keys()
        ), "need to provide preds for all models or use the default one"

        if self.evaluation is None:
            self.evaluation = LaMPEvaluation(
                single_gold_json_file_addr=self.task_output_file
            )

        for index, subscriber_name in enumerate(self.subscribers.keys()):
            if save_name_not_provided:
                curr_preds_save_name = os.path.join(
                    self.store_path,
                    f"LaMP_{self.task_id}_{self.task_type}_preds_{subscriber_name}_{self.suffix}.json",
                )
            else:
                curr_preds_save_name = preds_save_name[index]
            self.score[subscriber_name] = self.evaluation.evaluate_task(
                curr_preds_save_name, self.task_name
            )

    def subscribe(self) -> None:
        self.preds = dict()
        self.score = dict()

        with Progress() as prog:
            task_dict: Dict[str, Tuple[TaskID, Callable]] = dict()
            proc_worker: List[Thread] = list()
            task_worker: List[Thread] = list()
            worker_pipeline = Queue()

            for i in range(self.worker_count):
                curr_worker = Thread(
                    target=subscriber_proc_thread,
                    args=(
                        prog,
                        task_dict,
                        worker_pipeline,
                    ),
                )
                curr_worker.start()
                proc_worker.append(curr_worker)

            for subscriber_name, subscriber_func in self.subscribers.items():
                self.preds[subscriber_name] = labels(task=self.task_name, golds=list())
                curr_p_task = prog.add_task(
                    f"Processing Requests for {subscriber_name}", total=default_total
                )
                task_dict[subscriber_name] = (
                    curr_p_task,
                    subscriber_func,
                    self.preds[subscriber_name],
                )

                curr_q_task = prog.add_task(
                    f"Queuing Requests for {subscriber_name}", total=default_total
                )
                curr_worker = Thread(
                    target=subscriber_task_thread,
                    args=(
                        prog,
                        (subscriber_name, curr_q_task),
                        self.parsed_prompts,
                        worker_pipeline,
                    ),
                )
                curr_worker.start()
                task_worker.append(curr_worker)

            [worker.join() for worker in task_worker]

            for i in range(self.worker_count):
                worker_pipeline.put("finished")

            [worker.join() for worker in proc_worker]

            # feed prompt into subscriber
            for subscriber_name, subscriber_func in self.subscribers.items():
                # curr_preds = labels(task=self.task_name, golds=list())
                # subscriber_func(self.parsed_prompts, curr_preds)
                # self.preds[subscriber_name] = curr_preds

                if self.store_path is not None and self.store_path != "":
                    preds_save_name = os.path.join(
                        self.store_path,
                        f"LaMP_{self.task_id}_{self.task_type}_preds_{subscriber_name}_{self.suffix}.json",
                    )
                    os.makedirs(self.store_path, exist_ok=True)
                    with open(preds_save_name, "w", encoding="utf-8") as output:
                        json.dump(
                            self.preds[subscriber_name],
                            output,
                            cls=DTOEncoder,
                            indent=4,
                        )
                if self.evaluation is not None:
                    self.score[subscriber_name] = self.evaluation.evaluate_task(
                        preds_save_name, self.task_name
                    )

    def construct_prompt(self) -> None:
        self.parsed_prompts = parse_dataset_to_prompt(
            self.task_question_file,
            self.task_name,
            self.keyword_extraction,
        )

        if self.prompt_save_path is not None and self.prompt_save_path != "":
            prompt_save_name = os.path.join(
                self.prompt_save_path,
                f"LaMP_{self.task_id}_{self.task_type}_prompts_{self.suffix}.json",
            )
            with open(prompt_save_name, "w", encoding="utf-8") as output:
                json.dump(self.parsed_prompts, output, cls=DTOEncoder, indent=4)

    def run(self) -> None:
        self.construct_prompt()
        self.subscribe()

    def __call__(self, prompt_path: str = None) -> Any:
        if prompt_path is None or prompt_path == "":
            if self.prompt_save_path is None or self.prompt_save_path == "":
                return self.run()
            prompt_path = os.path.join(
                self.prompt_save_path,
                f"LaMP_{self.task_id}_{self.task_type}_prompts_{self.suffix}.json",
            )
        # cur_suffix = "_".join(prompt_path.split(".")[1].split("_")[-2:])
        # if cur_suffix != self.suffix:
        #     task_purpose = (
        #         "keyword-extraction"
        #         if self.keyword_extraction
        #         else "non-keyword-extraction"
        #     )
        #     file_purpose = (
        #         "keyword-extraction"
        #         if cur_suffix == "with_keyword"
        #         else "non-keyword-extraction"
        #     )
        #     raise RuntimeError(
        #         f"the file pass in is for {file_purpose}, but the task is for {task_purpose}"
        #     )
        if os.path.isfile(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as prompt_file:
                self.parsed_prompts = json.load(prompt_file)
            self.subscribe()
        else:
            self.run()
