from Utils.utils import loading_model_tokenizer ,loading_diff_datasets
import pandas as pd
from datasets import  Dataset 
from bert_score import score
from tqdm import tqdm 
import torch 

def extract_text(answer):
      if answer['text']:
        return answer['text'][0]
      else:
        return None  
      
def preprocess_data(data):
  if data is not None and isinstance(data, pd.DataFrame):
    df = dataset.to_pandas(data)
    df['extracted_text'] = df['answers'].apply(extract_text)
    df.dropna(subset=['extracted_text'], inplace=True)
    dataset = Dataset.from_pandas(df)
    return dataset   
  else:
    print("data is empty,please check again")

def analyze_indic_qa(text: str, question: str, answer: str, model, tokenizer):
    messages = [
        {"role": "system", "content": "Given a large body of text representing context, your task is to answer questions based on this context Answer the question in less than 6 words"},
        {"role": "user", "content": f"Context : {text} , Question : {question} ,Answer in hindi "},
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

    scores = score([model_answer], [answer], lang='hi')

    return {'Input Text': text, 'Actual answer': answer, 'Model answer': model_answer, 'bert score': scores}


def indic_qa_evaluate(dataset, model, tokenizer):
    data = []
    for context, question, answer in tqdm(zip(dataset['context'][0:20], dataset['question'][0:20], dataset['extracted_text'][0:20]), total=20): 
        if context is not None and question is not None:
            try:
                data.append(analyze_indic_qa(context, question, answer, model, tokenizer))
            except Exception as e:
                print(f"Error: {e}")
                continue
        else:
            print("Input text or label is None. Skipping...")

    return pd.DataFrame(data)

def average_(data):
        bert_scores = data['bert score']

        # Check if BertScore is present in the data
        if bert_scores:
            precision_list = [scores[0] for scores in bert_scores]
            recall_list = [scores[1] for scores in bert_scores]
            f1_list = [scores[2] for scores in bert_scores]

            # Compute the mean of precision, recall, and F1-score
            average_precision = torch.stack(precision_list).mean().item()
            average_recall = torch.stack(recall_list).mean().item()
            average_f1 = torch.stack(f1_list).mean().item()

            return average_precision, average_recall, average_f1
        else:
            print("Error: 'bert score' key not found in the data.")
            return None

      

if __name__ == "__main__":
    Model , Tokenizer = loading_model_tokenizer()
    dataset = loading_diff_datasets("ai4bharat/IndicQA",param='indicqa.hi',split='test')
    dataset = dataset.shuffle(seed=76)
    dataset = dataset.select(range(20))
    df = indic_qa_evaluate(dataset, Model, Tokenizer)
    df.to_csv("eval_indic_qa.csv")
    average_precision, average_recall, average_f1 = average_(df)
    print("Average BERTScore:")
    print("Precision:", average_precision)
    print("Recall:", average_recall)
    print("F1-score:", average_f1)
