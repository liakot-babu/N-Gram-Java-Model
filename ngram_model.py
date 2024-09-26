import pickle
import re
import json
from collections import Counter
from nltk.util import ngrams

def preprocess_code(code):
    # Remove comments
    code = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', code)
    return code

def tokenized_methods(data):
    tokenized_data = []
    for line in data:
        tokens = re.findall(r"[\w']+|[^\w\s']", line)
        tokenized_data.append(tokens)
    return tokenized_data

def save_tokens(data_path, token_path):
    datas = load_methods(data_path)
    datas = [preprocess_code(data) for data in datas]
    tokenized_datas = tokenized_methods(datas)

    tokens = []
    for data in tokenized_datas:
        tokens.extend(data)

    tokens = set(tokens)

    with open(token_path, 'wb') as f:
        pickle.dump(tokens, f)
        print('Tokenized data saved to {}'.format(token_path))

def load_methods(data_path):
    with open(data_path, 'r') as file:
        data = json.load(file)
    datas = []
    for folder in data:
        for file in folder['folder_data']:
            for method in file['file_data']:
                datas.append(method['method_data'])
    return datas

def save_ngram(data_path, ngram_path, n=2):
    datas = load_methods(data_path)
    datas = [preprocess_code(data) for data in datas]
    tokenized_datas = tokenized_methods(datas)

    ngram_data = []
    for data in tokenized_datas:
        ngram_list = list(ngrams(data, n))
        if len(ngram_list) < 1:
            continue
        ngram_data.extend(ngram_list)

    with open(ngram_path, 'wb') as f:
        pickle.dump(ngram_data, f)
        print(f"{n}-gram data saved to {ngram_path}")

def generate_ngram_prediction(token_path, ngram_path, n=2):
    with open(ngram_path, 'rb') as f:
        ngrams = pickle.load(f)

    with open(token_path, 'rb') as f:
        tokens = pickle.load(f)

    Counts = Counter(ngrams)

    max_prob = -1
    prediction = None

    context = ["public","void"]

    total_context_count = sum(count for ngram, count in Counts.items() if ngram[:n-1] == tuple(context))
    print(f"Total count for context {context}: {total_context_count}")

    for token in tokens:
        ngram_context = tuple(context + [token])
        ngram_count = Counts.get(ngram_context, 0)

        if total_context_count > 0:
            prob = ngram_count / total_context_count
            if prob > 0.1:
                print(f"ngram: {ngram_context}, prob: {prob}")
            if prob > max_prob:
                max_prob = prob
                prediction = token

    print(f"{' '.join(context)} {prediction}")

if __name__ == "__main__":
    data_path = "final_method_data.json"
    token_path = "tokens.pkl"
    ngram_path = "ngrams.pkl"
    n = 3

    save_tokens(data_path, token_path)
    save_ngram(data_path, ngram_path, n=n)
    generate_ngram_prediction(token_path, ngram_path, n=n)
