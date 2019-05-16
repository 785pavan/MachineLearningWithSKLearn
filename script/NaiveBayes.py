import os
import time
from collections import Counter

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


def make_dictionary(root_dir):
    all_words = []
    emails = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    for mail in emails:
        with open(mail) as m:
            for line in m:
                words = line.split()
                all_words += words
    dictionary = Counter(all_words)
    list_to_remove = list(dictionary)
    for item in list_to_remove:
        if not item.isalpha():
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary


def extract_features(mail_dir, dictionary):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), 3000))
    train_labels = np.zeros(len(files))
    count = 0
    docID = 0
    for fil in files:
        with open(fil) as fi:
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = words.count(word)
            train_labels[docID] = 0
            filepathTokens = fil.split('\\')
            lastToken = filepathTokens[len(filepathTokens) - 1]
            if lastToken.startswith("spmsg"):
                train_labels[docID] = 1
                count = count + 1
            docID = docID + 1
    return features_matrix, train_labels


if __name__ == "__main__":
    train_dir = "../train-mails"
    test_dir = "../test-mails"

    dictionary = make_dictionary(train_dir)
    feature_matrix, train_labels = extract_features(train_dir, dictionary)
    test_feature_matrix, test_labels = extract_features(test_dir, dictionary)

    # gaussian naiveBayes
    print("-------------------------------------------------------------------------------------------------")

    model = GaussianNB()
    Stime = time.time()
    model.fit(feature_matrix, train_labels)
    predicted_labels = model.predict(test_feature_matrix)
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(test_labels, predicted_labels)
    print("Gaussian NB accuracy: ", accuracy)
    print("Time taken: " + (-Stime + time.time()).__repr__())
    Stime = time.time()

    # multinormal naiveBayes
    print("-------------------------------------------------------------------------------------------------")

    model = MultinomialNB()
    model.fit(feature_matrix, train_labels)
    predicted_labels = model.predict(test_feature_matrix)
    accuracy = accuracy_score(test_labels, predicted_labels)
    print("Multinomial NB accuracy: ", accuracy)
    print("Time taken: " + (-Stime + time.time()).__repr__())
    Stime = time.time()

    # bernoulli naiveBayes
    print("-------------------------------------------------------------------------------------------------")

    model = BernoulliNB()
    model.fit(feature_matrix, train_labels)
    predicted_labels = model.predict(test_feature_matrix)
    accuracy = accuracy_score(test_labels, predicted_labels)
    print("Bernoulli NB accuracy: ", accuracy)
    print("Time taken: " + (-Stime + time.time()).__repr__())
    Stime = time.time()
    print("-------------------------------------------------------------------------------------------------")
