from Utils.utils import loading_model_tokenizer, loading_diff_datasets
import pandas as pd
from bert_score import score
import torch 
from tqdm import tqdm  


def analyze_summarization(text: str, summary: str, model, tokenizer):
    messages = [
        {"role": "system", "content": "Given a text you will provide a very short and clear and concise summary of it"},
        {"role": "user", "content": f"{text} Can you provide a summary of this, Answer in hindi"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    love = model.generate(input_ids=inputs, max_new_tokens=400, temperature=0.1, use_cache=True)
    output = tokenizer.decode(love[0], skip_special_tokens=True)
    model_summary = output.split("assistant\n")[1].strip()

    actual_summary = summary

    # Calculate BERTScore for the model summary against the actual summary
    scores = score([model_summary], [actual_summary], lang='hi')

    return {'Input Text': text, 'Actual Summary': actual_summary,
            'Model Summary': model_summary, 'BertScore': scores}


def evaluate_summarization(dataset, model, tokenizer, first_number: int, second_number: int):
    data = []
    for text, summary in tqdm(zip(dataset['article'][first_number:second_number], dataset['summary'][first_number:second_number]),total=100):
        if text is not None and summary is not None:
            try:
                data.append(analyze_summarization(text, summary, model, tokenizer))
            except Exception as e:
                print(f"Error: {e}")
                continue
        else:
            print("Input text or summary is None. Skipping...")

    return pd.DataFrame(data)

def average_(data):
    if not data.empty and 'bert score' in data:
        bert_scores = data['bert score']
        
        if not bert_scores.empty:
            precision_list = [scores[0] for scores in bert_scores]
            recall_list = [scores[1] for scores in bert_scores]
            f1_list = [scores[2] for scores in bert_scores]

            # Compute the mean of precision, recall, and F1-score
            average_precision = torch.stack(precision_list).mean().item()
            average_recall = torch.stack(recall_list).mean().item()
            average_f1 = torch.stack(f1_list).mean().item()

            return average_precision, average_recall, average_f1
        else:
            print("Error: 'bert score' column is empty.")
            return None
    else:
        print("Error: Invalid input data.")
        return None
   
        


if __name__ == "__main__":
    Model, Tokenizer = loading_model_tokenizer()
    dataset = loading_diff_datasets("Someman/hindi-summarization", split='train')
    dataset = dataset.shuffle(seed=76)
    dataset = dataset.select(range(20))
    df = evaluate_summarization(dataset, Model, Tokenizer, 0, 100)
    df.to_csv("eval_summarization.csv")
    average_precision, average_recall, average_f1 = average_(df)
    print("Average BERTScore:")
    print("Precision:", average_precision)
    print("Recall:", average_recall)
    print("F1-score:", average_f1)