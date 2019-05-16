from sklearn import tree
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
    model = tree.DecisionTreeClassifier()
    print("Training model")
    model.fit(features_matrix, labels)
    predicted_label = model.predict(test_feature_matrix)
    print("Finished classifying. accuracy score: ")
    print(accuracy_score(test_labels, predicted_label))
    print("-------------------------------------------------------------------------------------------------")
    model = tree.DecisionTreeClassifier(min_samples_split=40)
    print("Training model min_samples_split=40")
    model.fit(features_matrix, labels)
    predicted_label = model.predict(test_feature_matrix)
    print("Finished classifying. accuracy score: ")
    print(accuracy_score(test_labels, predicted_label))
    print("-------------------------------------------------------------------------------------------------")
    model = tree.DecisionTreeClassifier(criterion="entropy")
    print("Training model criterion=\"entropy\"")
    model.fit(features_matrix, labels)
    predicted_label = model.predict(test_feature_matrix)
    print("Finished classifying. accuracy score: ")
    print(accuracy_score(test_labels, predicted_label))
    print("-------------------------------------------------------------------------------------------------")
    model = tree.DecisionTreeClassifier(criterion="gini")
    print("Training model criterion=\"gini\"")
    model.fit(features_matrix, labels)
    predicted_label = model.predict(test_feature_matrix)
    print("Finished classifying. accuracy score: ")
    print(accuracy_score(test_labels, predicted_label))
