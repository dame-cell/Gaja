from datasets import load_dataset,concatenate_datasets,Dataset
import pandas as pd 


def load_datasets():
    dataset_sm = load_dataset("smangrul/hindi_instruct_v1", split='train')
    dataset_alp = load_dataset("NebulaByte/alpaca-gpt4-hindi-hinglish", split='train')
    dataset_hh = load_dataset("ai4bharat/indic-instruct-data-v0.1", 'hh-rlhf', split='hi')
    dataset_dol_hi = load_dataset("aaditya/databricks-dolly-15k-Hindi",split='train')
    dataset_dol_hin = load_dataset("aaditya/databricks-dolly-15k-Hinglish-Codemix",split='train')
    
    return dataset_sm, dataset_alp, dataset_hh,dataset_dol_hi, dataset_dol_hin


def preprocess_sm(dataset):
    """
    Pre-processes datasets_sm into an instruction dataset for easier handling.
    """
    system_content = []
    user_content = []
    assistant_content = []

    for row in dataset['messages']:
        system_content.append(next((msg['content'] for msg in row if msg['role'] == 'system'), ''))
        user_content.append(next((msg['content'] for msg in row if msg['role'] == 'user'), ''))
        assistant_content.append(next((msg['content'] for msg in row if msg['role'] == 'assistant'), ''))

    # Creating a DataFrame
    # Creating a DataFrame
    df = pd.DataFrame({'instruction': system_content, 'input': user_content, 'output': assistant_content})
    return df 


def preprocess_alp(dataset):
    """
    Pre-processes datasets_alp into an instruction dataset for easier handling.
    """
    data = dataset.to_pandas()
    df = data[:10_000]
    if len(df) == 10_000:
        combined_input_df = pd.concat([df['input'], df['input_hinglish']], ignore_index=True)
        combined_input_df.columns = ['combined_inputss']
        combined_output_df = pd.concat([df['output'], df['output_hinglish']], ignore_index=True)
        combined_output_df.columns = ['combined_output']
        result_df = pd.concat([combined_input_df, combined_output_df], axis=1)
        result_df.columns = ['input', 'output']

        return result_df


def preprocess_hh(dataset):
    """
    Pre-processes datasets_hh and dataset_anude into an instruction dataset for easier handling.
    """
    data = dataset.to_pandas()
    user_content = []
    assistant_content = []
    for row in data['messages']:
        user_content.append(next((msg['content'] for msg in row if msg['role'] == 'user'), ''))
        assistant_content.append(next((msg['content'] for msg in row if msg['role'] == 'assistant'), ''))

    # Creating a DataFrame
    df = pd.DataFrame({'input': user_content, 'output': assistant_content})
    return df

def preprocessing_dolly_hi(dataset, columns):
    """
    Pre-processes dataset_dol_hi, dataset_dol_hin into an instruction dataset for easier handling.
    """
    dff = dataset.to_pandas()
    df_new = dff[:5000]
    if len(df_new) == 5000:
        df_new[columns]
        return df_new

def merged_all_dataset():
    """
    Creates a new DataFrame with three columns: instruction, input, and output, by merging all preprocessed datasets.
    """
    dataset_sm, dataset_alp, dataset_hh, dataset_dol_hi, dataset_dol_hin = load_datasets()

    # Preprocess all datasets
    df_sm = preprocess_sm(dataset_sm)
    df_alp = preprocess_alp(dataset_alp)
    df_hh = preprocess_hh(dataset_hh)
    df_dolly_hi = preprocessing_dolly_hi(dataset_dol_hi, ["hindi_instruction", "hindi_input", "hindi_output"])
    df_dolly_hin = preprocessing_dolly_hi(dataset_dol_hin, ["codemix_instruction", "codemix_input", "codemix_output"])


    column_mapping_hi= {
      'hindi_instruction': 'instruction',
      'hindi_input': 'input',
      'hindi_output': 'output',
      
     }

    column_mapping_hin ={
      'codemix_instruction': 'instruction',
      'codemix_input': 'input',
      'codemix_output': 'output'
     }


    df_dolly_hi.rename(columns=column_mapping_hi, inplace=True)
    df_dolly_hin.rename(columns=column_mapping_hin, inplace=True)


    #convert to huggingface dataset object
    df_sm = Dataset.from_pandas(df_sm)
    df_alp = Dataset.from_pandas(df_alp)
    df_hh = Dataset.from_pandas(df_hh)
    df_dolly_hi = Dataset.from_pandas(df_dolly_hi)
    df_dolly_hin = Dataset.from_pandas(df_dolly_hin)

    dataset_cc = concatenate_datasets([df_sm, df_alp,df_hh,df_dolly_hi,df_dolly_hin])
    dff  = dataset_cc.to_pandas()
    dff.drop(columns=['en_instruction', 'en_input', 'en_output', 'id', 'en_category', 'hindi_category', 'codemix_category'],inplace=True)
    dataset = Dataset.from_pandas(dff)

    return dataset
  
if __name__ == "__main__":
  dataset = merged_all_dataset()
 
