# Gaja

We release Gaja , a Hindi/Hinglish chat model instruction finetuned on SarvamAI's OpenHathi model.

<p align="center">
  <img src="asset\Dariava.jpg" alt="Gajendra is a Hindi/Hinglish instruction-tuned model based on different instruct datasets." style="width: 45%; min-width: 300px;">
</p>


This repository contains the code for  "Gaja", a project focused on Fine-Tuning SarvamAI's OpenHathi model for Conversational task . which employs the LoRA + Unsloth methodology for efficient fine tuning. 

# Contents 
1) [Indic-Eval](#indic-eval)
2) [Usage-Note](#usage-note)
3) [Prompt](#prompt)
4) [Why-Bert-score](#Why-bert-score??)

If you appreciate this work and found it helpful, consider giving it a star ⭐️ on GitHub. Your support motivates me to continue improving and adding new features. Thank you for your encouragement!

# Indic-Eval
Performing zero-shot Evaluation on the model : 

| Task                   | # Samples | Accuracy | Precision | F1       | Recall   |   Metrics  | 
|------------------------|-----------|----------|-----------|----------|----------|------------|
| Indic-Sentiment Analysis     |    100   |  0.71    |   -       | 0.76     | -    |   Accuracy,F1 score          |
| Indic-QA Evaluation    |    50    | -    |   0.62      |  0.68    | 0.75   |       Bert Score          |
| Indic-Summarization Evaluation |  50    |  -       |   0.65      | 0.68     |0.72   |   Bert Score          |
| Indic-NLI                       | 50     | 0.24    |   -       | 0.17       |        |     Accuracy,F1 score              |


# Why-bert-score?? 

Alright, let's talk about why we're opting for BERTScore over traditional metrics like ROUGE for assessing summarization models. So, you're in the business of evaluating your model's prowess in summarization, right? Now, with the old-school ROUGE metric, you'd have the model whip up a summary and then compare it with the actual one. Now, that's all well and good if your aim is to have the model parrot back the data. But here's the kicker: Sometimes, the model or our fancy language model might not churn out the exact same summary as the actual one. But hold up! That doesn't mean it's missed the mark entirely. It could still hit the nail on the head, just in its own unique way. 

That's where BERTScore swoops in to save the day. Instead of getting hung up on verbatim matches, BERTScore digs deep into the semantics, weighing up how close the generated summary is to the real deal. So, even if the wording or structure ain't a carbon copy, if the gist is on point, BERTScore gives it a thumbs-up. And that, my friend, means a more nuanced and accurate evaluation of our summarization skills. With BERTScore in our corner, we're not just ticking boxes; we're truly getting to the heart of what makes a good summary.
BERTScore can indeed be a valuable metric for evaluating summarization models, especially when the focus is on capturing the semantic similarity between the generated and reference summaries. BERTScore considers the contextual embeddings of words and phrases, allowing it to capture subtle semantic nuances and similarities that might be missed by traditional metrics like ROUGE.

Beyond that, BERTScore has been shown to correlate well with human judgments of summary quality, making it a reliable metric for summarization evaluation.

But what's the con?

*  BERTScore relies on pre-trained language models like BERT, which may introduce biases or limitations inherent in the training data
*  BERTScore's performance may vary depending on the tokenization method used, especially when evaluating languages with complex morphologies or scripts. Inconsistent tokenization between the generated and reference text can affect the alignment and similarity scores.

 
# Prompt 

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
