import pickle
import re
import json
import nltk
import numpy as np
from tqdm import tqdm
from nltk.util import ngrams, bigrams
from collections import Counter
from collections import defaultdict, Counter

from sklearn.model_selection import train_test_split

from nltk.util import ngrams


def preprocess_code(code):
    # Remove comments
    code = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', code)
    return code


def tokenized_methods(data):
    tokenized_data = []
    for line in data:
        tokens = get_tokens(line)
        tokenized_data.extend(tokens)
    return tokenized_data


def get_tokens(line):
    tokens = re.findall(r"[\w']+|[^\w\s']", line)
    return tokens


def save_tokens(data_path, token_path):
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)
    tokenized_datas = tokenized_methods(datas)

    tokens = set(tokenized_datas)

    with open(token_path, 'wb') as f:
        pickle.dump(tokens, f)
        print('Tokenized data saved to {}'.format(token_path))


def add_padding(datas, n):
    res = []
    for data in datas:
        res.append('SOM ' * (n-1) + data + 'EOM')
    return res


def save_ngram(data_path, ngram_path, n=2):
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)
    datas = add_padding(datas, n)
    tokenized_datas = tokenized_methods(datas)

    ngram_data = list(ngrams(tokenized_datas, n))

    with open(ngram_path, 'wb') as f:
        pickle.dump(ngram_data, f)
        print(f"{n}-gram data saved to {ngram_path}")


def generate_ngram_prediction(test_data_path, token_path, ngram_path, n=2):
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    with open(ngram_path, 'rb') as f:
        ngram_data = pickle.load(f)

    with open(token_path, 'rb') as f:
        tokens = pickle.load(f)

    accuracies = []
    for test_method in test_data[:10]:
        test_method_tokens = get_tokens(test_method)
        test_ngram_data = list(ngrams(test_method_tokens, n))

        matching = 0

        for t_data in tqdm(test_ngram_data):
            context = tuple(t_data[:-1])
            true_value = t_data[n-1]

            prediction = get_prediction(ngram_data, context, tokens)

            if prediction == true_value:
                matching += 1

        accuracies.append(matching / len(test_method_tokens))

    print(f"Mean Accuracy for {n}-Gram Model: {np.mean(accuracies)}")


def get_prediction(ngrams, context, tokens):
    Counts = Counter(ngrams)

    max_prob = -1
    prediction = None
    total_context_count = sum(count for ngram, count in Counts.items() if ngram[:n - 1] == tuple(context))

    for token in tokens:
        ngram_context = tuple(list(context) + [token])
        ngram_count = Counts.get(ngram_context, 0)

        if total_context_count > 0:
            prob = ngram_count / total_context_count
            if prob > max_prob:
                max_prob = prob
                prediction = token
        else:
            prediction = "EOM"
            break

    return prediction

def load_methods(data_path):
    with open(data_path, 'r') as file:
        data = json.load(file)

    datas = []
    test_method_pattern = re.compile(r' test[a-zA-Z\d_$]* *\(.*', re.IGNORECASE)

    for folder in data:
        for file in folder['folder_data']:
            for method in file['file_data']:
                method_code = method['method_data']
                if not test_method_pattern.search(method_code.lower()):
                    datas.append(method_code.strip())
    return datas


def data_preprocess(data_path, train_data_path, test_data_path):
    datas = load_methods(data_path)
    datas = [preprocess_code(data) for data in datas]

    train_data, test_data = train_test_split(datas, test_size=.1, shuffle=True, random_state=42)

    # Save training data as a pickle file
    with open(train_data_path, 'wb') as train_file:
        pickle.dump(train_data, train_file)

    # Save test data as a pickle file
    with open(test_data_path, 'wb') as test_file:
        pickle.dump(test_data, test_file)


if __name__ == "__main__":
    data_path = "final_method_data.json"
    train_data_path = "train_data.pkl"
    test_data_path = "test_data.pkl"
    token_path = "tokens.pkl"
    ngram_path = "ngrams.pkl"

    data_preprocess(data_path, train_data_path, test_data_path)
    save_tokens(train_data_path, token_path)

    for n in range(2, 6):
        save_ngram(train_data_path, ngram_path, n=n)
        generate_ngram_prediction(test_data_path, token_path, ngram_path, n=n)
