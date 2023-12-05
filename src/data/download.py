# ===============================================================================================
# Author: Chang Zeng
# Multi-Thread Downloader for LaMP Dataset
# Dataset Stored in ./src/data/LaMP_<dataset index>_<dataset category>_<dataset type>.json
# For example `LaMP_1_dev_outputs.json` is the label data for validation task of dataset 1
# Source: https://lamp-benchmark.github.io/download
# ===============================================================================================

import json
import os
import threading
from typing import List

import requests
from rich.progress import Progress, TaskID


end_point = "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{index}/{category}/{category}_{type}.json"

storage = os.path.join("src", "data")
CHUNKSIZE = 5 * 1024 * 1024
default_total_step = 1000

msg_dict = dict()


def downloader(
    prog: Progress,
    prog_task: TaskID,
    file_name: str,
    index: int,
    category: str,
    curr_type: str,
):
    resp = requests.get(
        end_point.format(index=index, category=category, type=curr_type),
        stream=True,
    )
    if resp.status_code == 200:
        total_length = int(resp.headers["Content-length"])
        total_mbs = total_length / pow(1024, 2)
        prog.update(
            prog_task,
            description=f"Accepting {file_name} with length {total_mbs:.2f} mbs...",
        )
    else:
        prog.update(
            prog_task,
            description=f"{file_name} Does not Exist...",
            completed=default_total_step,
            advance=default_total_step,
        )
        return
    curr_length = 0
    content = ""

    for chunk in resp.iter_content(chunk_size=CHUNKSIZE):
        chunk: bytes
        if not chunk:
            continue
        content += chunk.decode("utf-8")
        curr_length += len(chunk)
        curr_mbs = int(curr_length) / pow(1024, 2)
        msg = (
            f"Accepted {file_name} with length {curr_mbs:.2f} / {total_mbs:.2f} mbs..."
        )
        advance = (len(chunk) / total_length) * default_total_step
        prog.update(prog_task, description=msg, advance=advance)
    prog.update(
        prog_task,
        description=f"Storing Content into File: {file_name}",
        completed=default_total_step,
        advance=default_total_step,
    )

    os.makedirs(storage, exist_ok=True)

    with open(os.path.join(storage, f"LaMP_{i}_{category}_{curr_type}.json"), "w") as f:
        content_json = json.loads(content)
        json.dump(
            content_json,
            f,
            indent=4,
        )
    prog.update(
        prog_task,
        description=f"{file_name} Process Finished...",
        completed=default_total_step,
        advance=default_total_step,
    )


def get_color(colors: List[str] = None) -> str:
    if colors is None:
        colors = ["red", "green", "cyan"]

    curr_index = 0
    while True:
        yield colors[curr_index % len(colors)]


def get_color_with_index(index: int = 0, colors: List[str] = None) -> str:
    if colors is None or not isinstance(colors, list):
        colors = ["red", "green", "cyan"]

    return colors[index % len(colors)]


if __name__ == "__main__":
    threads: List[threading.Thread] = []
    data_category = ["train", "dev", "test"]
    data_type = ["questions", "outputs"]

    with Progress() as progress:
        task_index = 0
        for i in range(3, 4, 1):
            for category in data_category:
                for curr_type in data_type:
                    file_name = f"LaMP_{i}_{category}_{curr_type}.json"
                    curr_task = progress.add_task(
                        f"[{get_color_with_index(task_index)}] Downloading {file_name}...",
                        total=default_total_step,
                    )
                    curr_thread = threading.Thread(
                        target=downloader,
                        args=(
                            progress,
                            curr_task,
                            file_name,
                            i,
                            category,
                            curr_type,
                        ),
                    )
                    curr_thread.start()
                    threads.append(curr_thread)
                    msg_dict[file_name] = f"Downloading {file_name}..."

                    task_index += 1

        while not progress.finished:
            try:
                pass
            except Exception as e:
                print("Catching Exception: ", e)
                break

        for thread in threads:
            thread.join()
