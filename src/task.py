from genericpath import isfile
import json
import os
from queue import Queue
from threading import Thread
import time
from typing import Any, Callable, Dict, List, Tuple

from src.utils import label

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
    api_key: str,
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
        curr_label = sub_func(p_id, prompt, api_key)
        curr_labels.golds.append(curr_label)
        prog.update(task, advance=100 / total_num)
        time.sleep(1)


def subscriber_task_thread(
    prog: Progress,
    task_info: Tuple[str, TaskID],
    parsed_prompts: Dict[str, List[str]],
    processed_prompts: List[str],
    pipeline: Queue,
):
    """
    task_info: tuple of subscriber name and taskID for progress
    processed_prompts: prompt ids that have already got response from previous runs

    send content to pipeline (tuple of subscriber name and (id, prompt) and total number)
    """
    sub_name, task = task_info
    total_num = len(parsed_prompts) - len(processed_prompts)
    for curr_prompt_id, curr_prompt in parsed_prompts.items():
        # ignore processed prompts
        if curr_prompt_id in processed_prompts:
            continue
        pipeline.put((sub_name, (curr_prompt_id, curr_prompt), total_num))
        prog.update(task, advance=default_total / total_num)


class LaMPTask:
    store_path: str  # where query, label, and results are stored
    extract_path: str  # where the results/lables are extract to for evaluation
    prompt_save_path: str
    preds_save_path: Dict[str, str]  # map subscriber and preds save name

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
        preds_save_path: Dict[str, str] = None,
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
            if preds_save_path is not None:
                assert len(preds_save_path) == len(
                    subscribers
                ), "Need a save path for all subscribers or use the default one!"
                self.preds_save_path = preds_save_path

        self.extract_path = extract_path
        self.prompt_save_path = prompt_save_path

        self.parsed_prompts = None
        self.preds = None
        self.score = None
        _, q_id, q_type, *_ = task_question_file.split(os.sep)[-1].split("_")
        _, o_id, o_type, *_ = task_output_file.split(os.sep)[-1].split("_")
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

    def evaluate(self, preds_save_name: Dict[str, str] = None):
        save_name_not_provided = preds_save_name is None or len(preds_save_name) == 0
        assert save_name_not_provided or len(preds_save_name) == len(
            self.subscribers.keys()
        ), "need to provide preds for all models or use the default one"

        if self.evaluation is None:
            self.evaluation = LaMPEvaluation(
                single_gold_json_file_addr=self.task_output_file
            )

        if self.score is None:
            self.score = dict()

        for _, subscriber_name in enumerate(self.subscribers.keys()):
            if save_name_not_provided:
                curr_preds_save_name = os.path.join(
                    self.store_path,
                    f"LaMP_{self.task_id}_{self.task_type}_preds_{subscriber_name}_{self.suffix}.json",
                )
            else:
                curr_preds_save_name = preds_save_name[subscriber_name]
            self.score[subscriber_name] = self.evaluation.evaluate_task(
                curr_preds_save_name, self.task_name
            )

    def fetch_complete_task(self, subscriber_name: str) -> List[str]:
        """
        return list of id for completed task (task id found in preds file)
        """
        output_save_name = self.preds_save_path[subscriber_name]
        if not os.path.exists(output_save_name) or not os.path.isfile(output_save_name):
            return []
        with open(output_save_name, "r", encoding="utf-8") as prev_preds:
            labels_json = json.load(prev_preds)
            assert labels_json["task"] == self.task_name.rstrip("_alt")
            prev_preds = labels(task=self.task_name, golds=list())
            prev_preds_list = []
            for task in labels_json["golds"]:
                prev_preds_list.append(task["id"])
                prev_preds.golds.append(label(**task))
            self.preds[subscriber_name] = prev_preds
            return prev_preds_list

    def subscribe(
        self,
        skip_eval: bool = False,
        continue_previous_run: bool = True,
        api_keys: List[str] = None,
    ) -> None:
        """
        skip_eval controls whether we conduct evaluation right after fetch preds from LM

        continue_previous_run controls whether we read previous preds file and remove those task from potential tasks list
        """
        assert (
            api_keys is None or len(api_keys) == self.worker_count
        ), "Either specify api key for every worker or just use the default one"
        self.preds = dict()
        self.score = dict()
        self.prev_preds = dict()

        # check whether we have path to store the preds (either created from designated store path or take given one)
        if (
            self.store_path is not None and self.store_path != ""
        ) or self.preds_save_path is not None:
            if self.store_path is not None:
                os.makedirs(self.store_path, exist_ok=True)
            if self.preds_save_path is None:
                self.preds_save_path = dict()

            for subscriber_name in self.subscribers.keys():
                if subscriber_name not in self.preds_save_path:
                    self.preds_save_path[subscriber_name] = os.path.join(
                        self.store_path,
                        f"LaMP_{self.task_id}_{self.task_type}_preds_{subscriber_name}_{self.suffix}.json",
                    )
                if continue_previous_run:
                    self.prev_preds[subscriber_name] = self.fetch_complete_task(
                        subscriber_name
                    )

        with Progress() as prog:
            task_dict: Dict[str, Tuple[TaskID, Callable]] = dict()
            proc_worker: List[Thread] = list()
            task_worker: List[Thread] = list()
            worker_pipeline = Queue()

            for i in range(self.worker_count):
                if api_keys is not None:
                    api_key = api_keys[i]
                else:
                    api_key = None
                curr_worker = Thread(
                    target=subscriber_proc_thread,
                    args=(
                        prog,
                        task_dict,
                        api_key,
                        worker_pipeline,
                    ),
                )
                curr_worker.start()
                proc_worker.append(curr_worker)

            # feed prompt into subscriber
            for subscriber_name, subscriber_func in self.subscribers.items():
                # could be initialized by fetch_complete_task
                if (
                    subscriber_name not in self.preds
                    or self.preds[subscriber_name] is None
                ):
                    self.preds[subscriber_name] = labels(
                        task=self.task_name, golds=list()
                    )
                curr_q_task = prog.add_task(
                    f"Queuing Requests for {subscriber_name}", total=default_total
                )
                curr_p_task = prog.add_task(
                    f"Processing Requests for {subscriber_name}", total=default_total
                )
                task_dict[subscriber_name] = (
                    curr_p_task,
                    subscriber_func,
                    self.preds[subscriber_name],
                )
                print(self.prev_preds[subscriber_name])
                curr_worker = Thread(
                    target=subscriber_task_thread,
                    args=(
                        prog,
                        (subscriber_name, curr_q_task),
                        self.parsed_prompts,
                        self.prev_preds[subscriber_name],
                        worker_pipeline,
                    ),
                )
                curr_worker.start()
                task_worker.append(curr_worker)

            [worker.join() for worker in task_worker]

            for i in range(self.worker_count):
                worker_pipeline.put("finished")

            [worker.join() for worker in proc_worker]

            # no need to store preds
            if self.preds_save_path is None:
                return self.preds

            for subscriber_name in self.subscribers.keys():
                # curr_preds = labels(task=self.task_name, golds=list())
                # subscriber_func(self.parsed_prompts, curr_preds)
                # self.preds[subscriber_name] = curr_preds
                curr_preds_save_name = self.preds_save_path[subscriber_name]
                with open(curr_preds_save_name, "w", encoding="utf-8") as output:
                    json.dump(
                        self.preds[subscriber_name],
                        output,
                        cls=DTOEncoder,
                        indent=4,
                    )
                if self.evaluation is not None and not skip_eval:
                    self.score[subscriber_name] = self.evaluation.evaluate_task(
                        curr_preds_save_name, self.task_name
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

    def load_prompts(self, prompt_path: str) -> None:
        if not os.path.isfile(prompt_path):
            raise FileNotFoundError("Try to parse prompts from non-existed file")
        with open(prompt_path, "r", encoding="utf-8") as prompt_file:
            self.parsed_prompts = json.load(prompt_file)

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
