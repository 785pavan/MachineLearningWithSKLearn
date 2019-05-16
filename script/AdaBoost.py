from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

import script.NaiveBayes as NB

if __name__ == "__main__":
    TRAIN_DIR = "../train-mails"
    TEST_DIR = "../test-mails"

    dictionary = NB.make_dictionary(TRAIN_DIR)

    print("Reading and processing emails from File.")
    features_matrix, labels = NB.extract_features(TRAIN_DIR, dictionary)
    test_feature_matrix, test_labels = NB.extract_features(TEST_DIR, dictionary)

    print("-------------------------------------------------------------------------------------------------")
    model = AdaBoostClassifier()
    print("Training Data")
    model.fit(features_matrix, labels)
    prediction = model.predict(test_feature_matrix)
    print("Finished classifying. accuracy score: ")
    print(accuracy_score(test_labels, prediction))
