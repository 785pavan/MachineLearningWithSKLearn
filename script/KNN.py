from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time
import script.NaiveBayes as NB

if __name__ == "__main__":
    TRAIN_DIR = "../train-mails"
    TEST_DIR = "../test-mails"

    dictionary = NB.make_dictionary(TRAIN_DIR)

    print("Reading and processing emails from File.")
    features_matrix, labels = NB.extract_features(TRAIN_DIR, dictionary)
    test_feature_matrix, test_labels = NB.extract_features(TEST_DIR, dictionary)

    print("-------------------------------------------------------------------------------------------------")
    model = KNeighborsClassifier(n_neighbors=3)
    print("Training model n_neighbors=3")
    Stime = time.time()
    model.fit(features_matrix, labels)
    predicted = model.predict(test_feature_matrix)
    print("Finished classifying. accuracy score: ")
    print(accuracy_score(test_labels, predicted))
    print("Time taken: " + (-Stime + time.time()).__repr__())
    Stime = time.time()
    print("-------------------------------------------------------------------------------------------------")
    model = KNeighborsClassifier(n_neighbors=5)
    print("Training model n_neighbors=5")
    model.fit(features_matrix, labels)
    predicted = model.predict(test_feature_matrix)
    print("Finished classifying. accuracy score: ")
    print(accuracy_score(test_labels, predicted))
    print("Time taken: " + (-Stime + time.time()).__repr__())
    Stime = time.time()
    print("-------------------------------------------------------------------------------------------------")
    model = KNeighborsClassifier(algorithm="ball_tree")
    print("Training model algorithm=\"ball_tree\"")
    model.fit(features_matrix, labels)
    predicted = model.predict(test_feature_matrix)
    print("Finished classifying. accuracy score: ")
    print(accuracy_score(test_labels, predicted))
    print("Time taken: " + (-Stime + time.time()).__repr__())
    Stime = time.time()
