import pickle
import re
import math
import json
import nltk
from nltk.util import ngrams, bigrams
from collections import Counter
from collections import defaultdict, Counter


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


def save_bigram(data_path, bigram_path):
    datas = load_methods(data_path)

    datas = [preprocess_code(data) for data in datas]
    tokenized_datas = tokenized_methods(datas)

    bigram = []
    for data in tokenized_datas:
        bigram_list = list(bigrams(data))
        if len(bigram_list) < 1:
            continue
        bigram.extend(bigram_list)

    with open(bigram_path, 'wb') as f:
        pickle.dump(bigram, f)
        print('Bigram data saved to {}'.format(bigram_path))


def generate_prediction(token_path, bigram_path):
    with open(bigram_path, 'rb') as f:
        bigrams = pickle.load(f)

    with open(token_path, 'rb') as f:
        tokens = pickle.load(f)

    # print(tokens)
    # print(Counter(bigrams))

    Counts = Counter(bigrams)

    max_prob = -1
    prediction = None

    context = "public"

    total_public_count = sum(count for (first, second), count in Counts.items() if first == context)
    print(f"Total public count: {total_public_count}")

    for token in tokens:
        bigram_count = Counts.get((context, token), 0)

        if total_public_count > 0:
            prob = bigram_count / total_public_count
            if prob > 0.1:
                print(f"token: {token}, prob: {prob}")
            if prob > max_prob:
                max_prob = prob
                prediction = token

    print(f"public {prediction}")


if __name__ == "__main__":
    data_path = "final_method_data.json"
    token_path = "tokens.txt"
    bigram_path = "bigrams.pkl"

    save_tokens(data_path, token_path)
    save_bigram(data_path, bigram_path)
    # generate_prediction(token_path, bigram_path)
