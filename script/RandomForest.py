from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

import script.NaiveBayes as NB

if __name__ == "__main__":
    Stime = time.time()
    TRAIN_DIR = "../train-mails"
    TEST_DIR = "../test-mails"

    dictionary = NB.make_dictionary(TRAIN_DIR)

    print("Reading and processing emails from File.")
    features_matrix, labels = NB.extract_features(TRAIN_DIR, dictionary)
    test_feature_matrix, test_labels = NB.extract_features(TEST_DIR, dictionary)

    print("-------------------------------------------------------------------------------------------------")
    model = RandomForestClassifier()
    print("Training data")
    model.fit(features_matrix, labels)
    prediction = model.predict(test_feature_matrix)
    print("Finished classifying. accuracy score: ")
    print(accuracy_score(test_labels, prediction))
    print("Time taken: " + (-Stime + time.time()).__repr__())
    Stime = time.time()
    print("-------------------------------------------------------------------------------------------------")
    model = RandomForestClassifier(criterion="entropy")
    print("Training data criterion=\"entropy\"")
    model.fit(features_matrix, labels)
    prediction = model.predict(test_feature_matrix)
    print("Finished classifying. accuracy score: ")
    print(accuracy_score(test_labels, prediction))
    print("Time taken: " + (-Stime + time.time()).__repr__())
    Stime = time.time()
    print("-------------------------------------------------------------------------------------------------")
    model = RandomForestClassifier(criterion="entropy", n_estimators=30)
    print("Training data criterion=\"entropy\" n_estimators=30")
    model.fit(features_matrix, labels)
    prediction = model.predict(test_feature_matrix)
    print("Finished classifying. accuracy score: ")
    print(accuracy_score(test_labels, prediction))
    print("Time taken: " + (-Stime + time.time()).__repr__())
    Stime = time.time()
