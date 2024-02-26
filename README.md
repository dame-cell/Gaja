# Gaja

We release Gaja , a Hindi/Hinglish chat model instruction finetuned on SarvamAI's OpenHathi model.
And Dairava which is a merge of Gaja and Ai4bharat instruct model "Airavata"


<p align="center">
  <img src="asset\Dariava.jpg" alt="Gajendra is a Hindi/Hinglish instruction-tuned model based on different instruct datasets." style="width: 45%; min-width: 300px;">
</p>


This repository contains the code for  "Gaja", a project focused on Instruct-Fine-tuning SarvamAI's OpenHathi model. which employs the LoRA methodology for efficient fine tuning. 

# Contents 
1) [Important Stuff to know](#important-stuff-to-know)
2) [Gaja Model-Based Variant](#gaja-model-based-variant)
3) [Dataset Information](#dataset-information)
4) [Gaja-Example output](#gaja-example-output)
5) [Dairava-Example output](#dairava-example-output)
6) [Gaja-Prompt Format](#gaja-prompt-format)
7) [Dairava-prompt Format](#dairava-prompt-format)
8) [Local Inference](#local-inference)
9) [Usage Note](#usage-note)


If you appreciate this work and found it helpful, consider giving it a star ⭐️ on GitHub. Your support motivates me to continue improving and adding new features. Thank you for your encouragement!

# Indic-Eval

| Task                   | # Samples | Accuracy | Precision | F1       | Recall   |   Metrics  | 
|------------------------|-----------|----------|-----------|----------|----------|------------|
| Sentiment Analysis     |    100   |  0.85    |   -       | 0.57     | -    |   Accuracy,F1 score          |
| Indic QA Evaluation    |    20    | -    |   0.62      |  0.68    | 0.75   |       Bert Score          |
| Summarization Evaluation |  20    |  -       |   0.68      | 0.71     |0.74    |   Bert Score          |



# Usage Note
It's important to note that the models have not undergone detoxification. Therefore, while they possess impressive linguistic capabilities, there is a possibility for them to generate content that could be deemed harmful or offensive. We urge users to exercise discretion and supervise the model's outputs closely, especially in public or sensitive applications.
