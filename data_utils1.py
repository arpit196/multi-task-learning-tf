import os
import wget
import tarfile
import re
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import pickle
import numpy as np

TRAIN_PATH = "dbpedia_csv/train.csv"
TEST_PATH = "dbpedia_csv/test.csv"


def download_dbpedia():
    dbpedia_url = 'https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz'

    wget.download(dbpedia_url)
    with tarfile.open("dbpedia_csv.tar.gz", "r:gz") as tar:
        tar.extractall()


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text


# TODO: Remove pickle
def build_word_dict():
    if not os.path.exists("word_dict.pickle"):
        train_df = pd.read_csv(TRAIN_PATH, names=["class", "title", "content"])
        train_df_snli=pd.read_csv("snli_1.0/snli_1.0_train.csv",names=["sentence1","sentence2","gold_label"])
        train_df_sts=pd.read_csv("snlitrain.csv",names=["sentence1","sentence2","gold_label"])
        contents = train_df["content"]
        contents.append()
        
        
        words = list()
        for content in contents:
            for word in word_tokenize(clean_str(content)):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, count in word_counter:
            if count > 1:
                word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    return word_dict


def build_dataset(step, word_dict, max_document_len):
    if step == "train":
        df = pd.read_csv(TRAIN_PATH, names=["class", "title", "content"])
        df_snli=pd.read_csv("snli_1.0/snli_1.0_train.csv",names=["sentence1","sentence2","gold_label"])
        df_sts=pd.read_csv("snlitrain.csv",names=["sentence1","sentence2","gold_label"])
    else:
        df = pd.read_csv(TEST_PATH, names=["class", "title", "content"])
        df_snli=pd.read_csv("snli_1.0/snli_1.0_test.csv",names=["sentence1","sentence2","gold_label"])
        df_sts=pd.read_csv("snlitrain.csv",names=["sentence1","sentence2","gold_label"])
        
    # Shuffle dataframe
    df = df.sample(frac=1)
    df_snli = df_snli.sample(frac=1)
    df_sts = df_sts.sample(frac=1)

    data = list(map(lambda d: word_tokenize(clean_str(d)), df["content"]))
    x = list(map(lambda d: ["<s>"] + d, data))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d[:max_document_len], x))
    x = list(map(lambda d: d + (max_document_len - len(d)) * [word_dict["<pad>"]], x))
    
    data1 = list(map(lambda d: word_tokenize(clean_str(d)), df_snli["sentence1"]))
    xnli1 = list(map(lambda d: ["<s>"] + d, data))
    xnli1 = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    xnli1 = list(map(lambda d: d[:max_document_len], x))
    xnli1 = list(map(lambda d: d + (max_document_len - len(d)) * [word_dict["<pad>"]], x))
    
    data2 = list(map(lambda d: word_tokenize(clean_str(d)), df_snli["sentence2"]))
    xnli2 = list(map(lambda d: ["<s>"] + d, data))
    xnli2 = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    xnli2 = list(map(lambda d: d[:max_document_len], x))
    xnli2 = list(map(lambda d: d + (max_document_len - len(d)) * [word_dict["<pad>"]], x))
    
    xsts1  list(map(lambda d: word_tokenize(clean_str(d)), df_sts["sentence1"]))
    xsts1 = list(map(lambda d: ["<s>"] + d, data))
    xsts1 = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    xsts1 = list(map(lambda d: d[:max_document_len], x))
    xsts1 = list(map(lambda d: d + (max_document_len - len(d)) * [word_dict["<pad>"]], x))
    
    xsts2 = list(map(lambda d: word_tokenize(clean_str(d)), df_sts["sentence2"]))
    xsts2 = list(map(lambda d: ["<s>"] + d, data))
    xsts2 = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    xsts2 = list(map(lambda d: d[:max_document_len], x))
    xsts2 = list(map(lambda d: d + (max_document_len - len(d)) * [word_dict["<pad>"]], x))
    
    lm_y = list(map(lambda d: d + ["</s>"], data))
    lm_y = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), lm_y))
    lm_y = list(map(lambda d: d[:max_document_len], lm_y))
    lm_y = list(map(lambda d: d + (max_document_len - len(d)) * [word_dict["<pad>"]], lm_y))
    
    clf_nli = list(map(lambda d: d - 1, list(df["gold_label"])))
    
    clf_sts = list(map(lambda d: d - 1, list(df[""])))

    clf_y = list(map(lambda d: d - 1, list(df["class"])))

    return x, xnli1, xnli2, xsts1, xsts2, lm_y, clf_y, clf_sts, clf_nli


def batch_iter(inputs, lm_outputs, clf_outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    lm_outputs = np.array(lm_outputs)
    clf_outputs = np.array(clf_outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], lm_outputs[start_index:end_index], clf_outputs[start_index:end_index]
