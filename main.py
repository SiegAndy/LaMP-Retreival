from src.task import LaMPTask

if __name__ == "__main__":
    task = LaMPTask(
        "./src/data/LaMP_11_train_questions.json",
        "./src/data/LaMP_11_train_outputs.json",
    )
    score = task.run()
    print(score)
