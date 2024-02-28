from Utils.utils import loading_model_tokenizer ,loading_diff_datasets
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm  
import torch 




def analyze_bool_qa(passage: str, question: str, answer: str, model, tokenizer):
    messages = [
        {"role": "system", "content": "Given a large body of text representing context, your task is to tell the user whether the question is True or False based on this context "},
        {"role": "user", "content": f"Context : {passage} , Question : {question} ,Answer in English "},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    love = model.generate(input_ids=inputs, max_new_tokens=400, temperature=0.1, use_cache=True)
    output = tokenizer.decode(love[0], skip_special_tokens=True)
    model_answer = output.split("assistant\n")[1].strip()  
    
    if "yes" in model_answer.lower():
        model_answer = "True"
    elif "no" in model_answer.lower():
        model_answer = "False"
    else:
        model_answer = "Unknown"

    return {'Input Text': passage, 'Actual answer': answer, 'Model answer': model_answer}



def bool_qa_evaluate(dataset, model, tokenizer,first_number, second_number):
    data = []
    for passage, question, answer in tqdm(zip(dataset['itv2 hi passage'][first_number:second_number], dataset['itv2 hi question'][first_number:second_number], dataset['answer'][first_number:second_number]), total=50): 
        if passage is not None and question is not None:
            try:
                data.append(analyze_bool_qa(passage, question, answer, model, tokenizer))
            except Exception as e:
                print(f"Error: {e}")
                continue
        else:
            print("Input text or label is None. Skipping...")

    return pd.DataFrame(data)

if __name__ == "__main__":
    dataset = loading_diff_datasets("ai4bharat/boolq-hi",split='train')
    model,tokenizer = loading_model_tokenizer()
    dataset = dataset.shuffle(seed=76)
    dataset = dataset.select(range(50))
    df = bool_qa_evaluate(dataset, model, tokenizer, first_number=0, second_number=50)
    
    accuracy = accuracy_score(df['Actual answer'], df['Model answer'])  
    f1 = f1_score(df['Actual answer'], df['Model answer'], average='weighted')

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    df.to_csv("eval_indic-bool_qa.csv")



