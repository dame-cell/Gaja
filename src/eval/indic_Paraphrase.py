import pandas as pd
from tqdm import tqdm
from evaluate import load
from sklearn.metrics import accuracy_score, f1_score ,precision_score
from Utils.utils import loading_model_tokenizer 
from datasets import load_dataset 

def analyze_paraphrase(input_text: str, choice1: str, choice2:str, model, tokenizer):
    
    messages = [
        {"role": "user", "content": f"{input_text} , {choice1}\n2. Second choice - {choice2}\n which translation do you think is correct first one or second one. Answer in english?"}
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

    if "first" in model_answer:
        return 0 
    elif "second" in model_answer:
        return 1
    else:
        return -1

    

def eval_paraphrase(dataset, model, tokenizer):
    data = []
    for text, choice1 , choice2,label in tqdm(zip(dataset['english'],dataset['sentence1'],dataset['sentence2'], dataset['label']), total=len(dataset)):
        if text is not None and label is not None:
            model_sentiment = analyze_paraphrase(text,choice1,choice2, model, tokenizer)
            data.append({'Input Text': text, 'Actual answer': label,  'Model answer': model_sentiment})
        else:
            print("Input text or label is None. Skipping...")
    return pd.DataFrame(data)


if __name__ == "__main__":
    model ,tokenizer = loading_model_tokenizer()
    dataset = load_dataset('ai4bharat/IndicXParaphrase','hi',split='test')
    shuffled_dataset = dataset.shuffle(seed=76)
    
    # Select a subset of data
    selected_dataset = shuffled_dataset.select(range(100))
    
    df = eval_paraphrase(selected_dataset, model, tokenizer)
    accuracy = accuracy_score(df['Actual answer'], df['Model answer'])  
    precision = precision_score(df['Actual answer'], df['Model answer']) 
    f1 = f1_score(df['Actual answer'], df['Model answer'], average='weighted')

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:",precision)
