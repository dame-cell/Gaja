from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch 



def loading_model_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained("damerajee/Gaja-vv1")
        model = AutoModelForCausalLM.from_pretrained("damerajee/Gaja-vv1", load_in_4bit=True)
        return model, tokenizer
    except OSError as e:
        print(f"Error: Unable to load the model or tokenizer. {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None
      
def loading_diff_datasets(dataset_name, split, param=None):
    try:
        if param is None:
            dataset = load_dataset(dataset_name, split)
        else:
            dataset = load_dataset(dataset_name, split, **param)
        return dataset 
    except OSError as e:
        print(f"Error: Unable to load the model or tokenizer. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
      
      
