# [Convergence of Keyphrase Extraction and Personalized Large Language Models](https://github.com/SiegAndy/LaMP-Retreival)
In our everyday interactions with search engines like Google, the use of keywords or phrases has become primarily used for locating the desired information. However, when advanced language models emerged, our approach to formulating queries evolved to constructing complete sentences, allowing for more precise expression of our needs. This proposed an important question: can these large language models effectively generate relevant responses when we incorporate key phrases into our prompts? Furthermore, in instances where personalization is a crucial factor, models such as ChatGPT utilize the conversation history to not only address current queries but also interact with past conversations. But is it capable of achieving comparable outcomes by utilizing personalized key phrases to our prompts? To explore these questions, this paper investigates the potential of incorporating key phrases into prompts to enhance the performance of large language models, emphasizing the importance of personalization and its impact on query understanding and response generation.

# Scripts
All code are under `./code` directory. Note that current directory structure in zip file is not my working folder structure. If you want to reproduce the results, please refer to [LaMP-Retreival](https://github.com/SiegAndy/LaMP-Retreival) or the folder structure section.


<li>  <code>selector.ipynb</code>: jupyter notebook file that randomly sample users from all possible users.

<li> <code>model_eval_task_1.ipynb</code>: jupyter notebook file that feed profiles/prompts into specified models for dataset LaMP 1.

<li>  <code>model_eval_task_1.ipynb</code>: jupyter notebook file that feed profiles/prompts into specified models for dataset LaMP 2.

<li>  <code>model_eval_clarification.ipynb</code>: jupyter notebook file that extract predictions/labels information. 

<br>

#### Note! Multi-threading is implemented to improve the running time. However, it might quickly reach the rate-limit of Huggingface's Inference API. You can set number of worker to 1 to avoid such situation.


# Package & Requirement
<li> Python Version 3.11.5
<li> Install packages via <code>pip install -r requirement.txt</code>.
<li> a <code>.config</code> file is expected to be presented in root directory. It requires following fields:

```
OPENAI_API_KEY=
HUGGING_FACE_KEY=
HUGGING_FACE_KEY_1=
HUGGING_FACE_KEY_2=
HUGGING_FACE_KEY_3=
HUGGING_FACE_KEY_4=
```



# Folder Stucture

Here are my working folder structure (not the one presented in zip file)

```
root
│
└─── README.md
│
└─── selector.ipynb
│
└─── model_eval_task_1.ipynb
│
└─── model_eval_task_2.ipynb
│
└─── model_eval_clarification.ipynb
│
└─── requirement.txt
│
└─── .config
│
└─── src
     │ 
     └─── data
     │      │ 
     │      └─── LaMP_1_train_questions.json
     │      │ 
     │      └─── LaMP_1_train_outputs.json
     │      │ 
     │      └─── LaMP_1 (Folder contains extracted profiles, prompts, and predictions from models)
     │      │ 
     │      └─── LaMP_2_train_questions.json
     │      │ 
     │      └─── LaMP_2_train_outputs.json
     │      │ 
     │      └─── LaMP_2 (Folder contains extracted profiles, prompts, and predictions from models)
     │ 
     └─── models
     │      │ 
     │      └─── BM25.py
     │      │ 
     │      └─── model.py
     │      │ 
     │      └─── TextRank.py
     │ 
     └─── utils (Folder contains utility classes/functions/variables)
     │ 
     └─── tokenization.py
     │ 
     └─── extraction.py
     │ 
     └─── task.py
     │ 
     └─── evaluation.py
```