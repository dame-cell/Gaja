# Gaja

We release Gaja , a Hindi/Hinglish chat model instruction finetuned on SarvamAI's OpenHathi model.

<p align="center">
  <img src="asset\Dariava.jpg" alt="Gajendra is a Hindi/Hinglish instruction-tuned model based on different instruct datasets." style="width: 45%; min-width: 300px;">
</p>


This repository contains the code for  "Gaja", a project focused on Fine-Tuning SarvamAI's OpenHathi model for Conversational task . which employs the LoRA + Unsloth methodology for efficient fine tuning. 

# Contents 
1) [Information](#information)
2) [Indic-Eval](#indic-eval)
3) [English-eval](#english-eval)
4) [Prompt-Format](#prompt-format)
5) [Usage-Note](#usage-note)
5) 

If you appreciate this work and found it helpful, consider giving it a star ⭐️ on GitHub. Your support motivates me to continue improving and adding new features. Thank you for your encouragement!

# Information 
* The model was fine-tuned on only 1k samples
* The

  
# Indic-Eval
Conducting a comprehensive zero-shot evaluation across various tasks, followed by the averaging of all scores, provides a holistic assessment of the model's performance.

| Task                   | # Samples | Accuracy | Precision | F1   | Recall | BLEU Score | Metrics                    |
|------------------------|-----------|----------|-----------|------|--------|------------|----------------------------|
| Indic-Sentiment Analysis | 100      | 0.71     | -        | 0.76 | -     | -          | Accuracy, F1 score       |
| Indic-QA Evaluation     | 50       |  -       | 0.62      | 0.68 | 0.75   | -          | Bert Score               |
| Indic-NLI               | 50       | 0.24     | -        | 0.17 | -     | -          | Accuracy, F1 score       |
| Indic-Paraphrase       | 500       | 0.52      | 0.49       | 0.48  | -     | 0.71       |  Accuracy, F1 score        |

# English-Eval

Model name| Average  | ARC | HellaSwag | MMLU | TruthfulQA   | Winogrande | GSM8K|      
|-------|------------------------|-----------|----------|-----------|------|--------|------------|       
| [damerajee/Gaja-v1.00](https://huggingface.co/damerajee/Gaja-v1.00)| 	47.69 | 52.82 |    76.31  |     40.83   | 44.64	| 	 70.64       |    0.91   |  
| [manishiitg/open-aditi-hi-v2](https://huggingface.co/manishiitg/open-aditi-hi-v2) | 	59.31 | 59.39 |  82.01   |   61.41     | 45.84 	| 	77.19        |    30.02  |    
| [ai4bharat/Airavata](https://huggingface.co/ai4bharat/Airavata) | 	45.52 | 46.5 |    69.26  |     43.9   | 40.62	| 	 68.82       |    4.02   |      

# Prompt-Format

The prompt for the Model without system prompt 
```python
<|im_start|>user
{}<|im_end|> 
<|im_start|>assistant
{}<|im_end|> 
```
The prompt for the Model with system prompt 
```python
|im_start|>system
{}<|im_end|> 
<|im_start|>user
{}<|im_end|> 
<|im_start|>assistant
{}<|im_end|> 
```

# Usage-Note
It's important to note that the models have not undergone detoxification. Therefore, while they possess impressive linguistic capabilities, there is a possibility for them to generate content that could be deemed harmful or offensive. We urge users to exercise discretion and supervise the model's outputs closely, especially in public or sensitive applications.
