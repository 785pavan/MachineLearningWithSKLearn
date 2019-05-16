import cPickle
import gzip
def load(file_name):
    # load the model
    stream = gzip.open(file_name, "rb")
    model = cPickle.load(stream)
    stream.close()
    return model
def save(file_name, model):
    # save the model
    stream = gzip.open(file_name, "wb")
    cPickle.dump(model, stream)
    stream.close()
#To save
save("/tmp/features_matrix", features_matrix)
save("/tmp/labels", labels)
save("/tmp/test_feature_matrix", test_feature_matrix)
save("/tmp/test_labels", test_labels)
#To load
features_matrix = load("/tmp/features_matrix")
labels = load("/tmp/labels")
test_feature_matrix = load("/tmp/test_feature_matrix")
test_labels = load("/tmp/test_labels")