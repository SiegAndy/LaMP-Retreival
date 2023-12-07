import os

config_path = ".config"

default_text_rank_window_size = 2
default_top_k_keywords = 5
default_data_path = os.path.join("src", "data")
default_prompt_path = os.path.join(default_data_path, "prompt")
default_extract_path = os.path.join(default_data_path, "extracts")


task_2_categories = [
    "women",
    "religion",
    "politics",
    "style & beauty",
    "entertainment",
    "culture & arts",
    "sports",
    "science & technology",
    "travel",
    "business",
    "crime",
    "education",
    "healthy living",
    "parents",
    "food & drink",
]
task_2_categories_str = 'Categories: ["women", "religion", "politics", "style & beauty", "entertainment", "culture & arts", "sports", "science & technology", "travel", "business", "crime", "education", "healthy living", "parents", "food & drink"]'
