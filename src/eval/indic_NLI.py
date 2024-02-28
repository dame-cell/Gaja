import pandas as pd
from Utils.utils import loading_model_tokenizer ,loading_diff_datasets

from datasets import load_dataset, Dataset 
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm 
import torch 


def analyze_indic_qa(premise: str, hypothesis : str,model, tokenizer):
    messages = [
        {"role": "system", "content": "Given a premise and a hypothesis, your task is to determine the relationship between them: Entailment, Contradiction, or Neutral."},
        {"role": "user", "content": f"Premise: {premise}, Hypothesis: {hypothesis}, What is the relationship? (Entailment, Contradiction, Neutral)"}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    love = model.generate(input_ids=inputs, max_new_tokens=400, temperature=0.1, use_cache=True)
    output = tokenizer.decode(love[0], skip_special_tokens=True)
    model_answer = output.split("assistant\n")[1].strip().lower()  

    if "neutral" in model_answer:
        return 1
    elif "entailment" in model_answer:
        return 2
    elif "contradiction" in model_answer:
        return 0
    else:
        raise ValueError("Unknown label:", model_answer)


def indic_qa_evaluate(dataset, model, tokenizer, first_number, second_number):
    data = []
    for premise, hypothesis, label in tqdm(zip(dataset['premise'][first_number:second_number], dataset['hypothesis'][first_number:second_number], dataset['label'][first_number:second_number]), total=50): 
        if premise is not None and hypothesis is not None:
            try:
                model_answer = analyze_indic_qa(premise, hypothesis, model, tokenizer)  # Pass tokenizer here
                data.append({'Premise': premise, 'Hypothesis': hypothesis, 'Actual Label': label, 'Model Label': model_answer})
            except Exception as e:
                print(f"Error: {e}. Skipping...")
    return pd.DataFrame(data)


if __name__ == "__main__":
    model,tokenizer = loading_model_tokenizer()
    dd = loading_diff_datasets("Divyanshu/indicxnli",param='hi', split='test')
    df = indic_qa_evaluate(dd, model, tokenizer, first_number=0, second_number=50)
    df.to_csv("eval_indic_qa.csv")

    accuracy = accuracy_score(df['Actual Label'], df['Model Label'])  
    f1 = f1_score(df['Actual Label'], df['Model Label'], average='weighted')

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)

    df.to_csv("eval_sentiment_analysis.csv")