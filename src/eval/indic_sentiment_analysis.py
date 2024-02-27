from Utils.utils import loading_model_tokenizer ,loading_diff_datasets
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm  

def analyze_sentiment(text: str, model, tokenizer):
    messages = [
        {"role": "system", "content": "Given a text you will predict its Sentiment"},
        {"role": "user", "content": f"{text} Can you tell me if this text is Positive or Negative, Answer in English"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    love = model.generate(input_ids=inputs, max_new_tokens=400, temperature=0.1, use_cache=True)
    output = tokenizer.decode(love[0], skip_special_tokens=True)
    model_sentiment = output.split("assistant\n")[1].strip()  # Get the sentiment prediction

    # Extract positive or negative sentiment from the model response
    if "positive" in model_sentiment.lower():
        model_sentiment = "Positive"
    elif "negative" in model_sentiment.lower():
        model_sentiment = "Negative"
    else:
        model_sentiment = "Unknown"

    return model_sentiment


def eval_sentiment_analysis(dataset, model, tokenizer):
    data = []
    for text, label in tqdm(zip(dataset['INDIC REVIEW'], dataset['LABEL']), total=len(dataset),total=100):
        if text is not None and label is not None:
            model_sentiment = analyze_sentiment(text, model, tokenizer)
            data.append({'Input Text': text, 'Actual Sentiment': label, 'Model Sentiment': model_sentiment})
        else:
            print("Input text or label is None. Skipping...")
    return pd.DataFrame(data)


  
if __name__ == "__main__":
    Model, Tokenizer = loading_model_tokenizer()
    dataset = loading_diff_datasets("ai4bharat/IndicSentiment", split='validation', param='translation-hi')
    dataset = dataset.shuffle(seed=76)
    dataset = dataset.select(range(100))
    df = eval_sentiment_analysis(dataset, Model, Tokenizer)
    accuracy = accuracy_score(df['Actual Sentiment'], df['Model Sentiment'])  
    f1 = f1_score(df['Actual Sentiment'], df['Model Sentiment'], average='weighted')

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)

    df.to_csv("eval_sentiment_analysis.csv")


