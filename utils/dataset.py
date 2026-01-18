import os 
import pandas as pd

def load_dataset(folder_path: str = "./datasets", header=None, names:list=["prompt"]):
    """
    folder_path: Loads all csv files in folder_path as a single dataset
    """
    data = pd.DataFrame()

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=header, names=names)
            data = pd.concat([data, df], ignore_index=True)

    return data

def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]

def get_harmful_instructions():
    dataset = load_dataset('mlabonne/harmful_behaviors')
    return reformat_texts(dataset['train']['text']), reformat_texts(dataset['test']['text'])

def get_harmless_instructions():
    dataset = load_dataset('mlabonne/harmless_alpaca')
    return reformat_texts(dataset['train']['text']), reformat_texts(dataset['test']['text'])
