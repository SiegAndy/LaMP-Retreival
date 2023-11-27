import subprocess
from src.extraction import read_dataset


if __name__ == "__main__":
    parsed_result = read_dataset("./src/data/LaMP_11_train_questions.json", 1)
    # parsed_result = read_dataset("./src/data/LaMP_22_train_questions.json", 2)
    # print(parsed_result)
    # "Which category does this article relate to among the following categories? Just answer with the category name without further explanation. categories: [women, religion, politics, style & beauty, entertainment, culture & arts, sports, science & technology, travel, business, crime, education, healthy living, parents, food & drink] article: Beets are a nutritional powerhouse -- they cleanse the body, are chock-full of vitamins, minerals and antioxidants, and are a great source of energy.",
    # import os
    # def addToClipBoard(text):
    #     command = 'echo ' + text.strip() + '| clip'
    #     subprocess.check_call(command, shell=True)

    # # Example
    # addToClipBoard('penny laneaaaa')