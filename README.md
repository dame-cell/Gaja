# Gajendra

We release Gajendra, a Hindi/Hinglish chat model instruction finetuned on SarvamAI's OpenHathi model.

<p align="center" width="100%">
    <img src="asset\gajendra.jpg" alt="Gajendra is a Hindi/Hinglish instruction-tuned model based on different instruct datasets." style="width: 50%; min-width: 600px; display: block; margin: auto;">
</p>



This repository contains the code for  "Gajendra", a project focused on Instruct-Fine-tuning SarvamAI's OpenHathi model. which employs the LoRA methodology for efficient fine tuning. 

# Important Stuff to know 

* The total rows of the instruct dataset contains about 50k rows 
* The entire finetune process was done in a free goggle colab using the t4 gpu 
* The entire dataset was jsut a combination of different dataset from hugging face hub
* The total amount of dataset used here is - 5 

| dataset name | Author | Link |
|----------|----------|----------|
| hindi_instruct_v | smangrul |[hindi-instruct-v](https://huggingface.co/datasets/smangrul/hindi_instruct_v1)|
| alpaca-gpt4-hindi-hinglish | NebulaByte | [alpaca-gpt4-hindi-hinglish](https://huggingface.co/datasets/NebulaByte/alpaca-gpt4-hindi-hinglish) |
| indic-instruct-data-v0.1 | ai4bharat |[indic-instruct-data-v0.1](https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1) |
| databricks-dolly-15k-Hindi| aaditya|[databricks-dolly-15k-Hindi](https://huggingface.co/datasets/aaditya/databricks-dolly-15k-Hindi)|
| databricks-dolly-15k-Hinglish-Codemix | aaditya|[databricks-dolly-15k-Hinglish-Codemix](https://huggingface.co/datasets/aaditya/databricks-dolly-15k-Hinglish-Codemix)|