from sklearn import svm
from sklearn.metrics import accuracy_score
import time
import script.NaiveBayes as NB

if __name__ == "__main__":
    TRAIN_DIR = "../train-mails"
    TEST_DIR = "../test-mails"
    dictionary = NB.make_dictionary(TRAIN_DIR)

    print("Reading and processing emails from File.")
    features_matrix, labels = NB.extract_features(TRAIN_DIR, dictionary)
    test_feature_matrix, test_labels = NB.extract_features(TEST_DIR, dictionary)

    print("Kernel = linear, C = 1, gemma = auto")
    model = svm.SVC(kernel='linear')
    print("Training Model")
    # train Model
    Stime = time.time()
    model.fit(features_matrix, labels)
    predicted_labels = model.predict(test_feature_matrix)
    print("FINISHED Classifying. accuracy score : ")
    print(accuracy_score(test_labels, predicted_labels))
    print("Time taken: " + (-Stime + time.time()).__repr__())
    Stime = time.time()
    print("-------------------------------------------------------------------------------------------------")
    model = svm.SVC(kernel='rbf', C=1)
    model.fit(features_matrix, labels)
    predicted_labels = model.predict(test_feature_matrix)
    print("FINISHED Classifying. accuracy score : ")
    print(accuracy_score(test_labels, predicted_labels))
    print("Time taken: " + (-Stime + time.time()).__repr__())
    Stime = time.time()
    print("-------------------------------------------------------------------------------------------------")
