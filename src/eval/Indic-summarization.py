from Utils.utils import loading_model_tokenizer, loading_diff_datasets
import pandas as pd
from bert_score import score



def evaluate_summarization(dataset, model, tokenizer, first_number: int, second_number: int):
    result = []
    data = []

    for text, summary in zip(dataset['article'][first_number:second_number], dataset['summary'][first_number:second_number]):
        if text is not None and summary is not None:
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

            # Append the results to the list as dictionaries
            data.append({'Input Text': text, 'Actual Summary': actual_summary,
                         'Model Summary': model_summary, 'BertScore': scores})
        else:
            print("Input text or summary is None. Skipping...")

    return pd.DataFrame(data)


if __name__ == "__main__":
    Model, Tokenizer = loading_model_tokenizer()
    dataset = loading_diff_datasets("Someman/hindi-summarization", split='train')
    dataset = dataset.shuffle(seed=76)
    dataset = dataset.select(range(20))
    df = evaluate_summarization(dataset, Model, Tokenizer, 0, 20)
    df.to_csv("eval_summarization.csv")
