from models.doc2vec_model import doc2VecModel
from models.classifier_model import classifierModel

import os
import logging
import inspect

import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
project_dir_path = os.path.dirname(os.path.abspath(base_file_path))
data_path = os.path.join(project_dir_path, 'data')
default_classifier = os.path.join(
    project_dir_path, 'classifiers', 'logreg_model.pkl')
default_doc2vec = os.path.join(project_dir_path, 'classifiers', 'd2v.model')
default_dataset = os.path.join(data_path, 'dataset.csv')


class TextClassifier():

    def __init__(self):
        super().__init__()
        self.d2v = doc2VecModel()
        self.classifier = classifierModel()
        self.dataset = None

    def read_data(self, filename):
        filename = os.path.join(data_path, filename)
        self.dataset = pd.read_csv(filename, header=0, delimiter=",")

    def prepare_all_data(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.dataset.review, self.dataset.sentiment, random_state=0,
            test_size=0.1)
        x_train = doc2VecModel.label_sentences(x_train, 'Train')
        x_test = doc2VecModel.label_sentences(x_test, 'Test')
        all_data = x_train + x_test
        return x_train, x_test, y_train, y_test, all_data

    def prepare_test_data(self, sentence):
        x_test = doc2VecModel.label_sentences(sentence, 'Test')
        return x_test

    def train_classifier(self, type_file, target_file, original_file):
        x_train, x_test, y_train, y_test, all_data = self.prepare_all_data()
        self.d2v.initialize_model(all_data)
        self.d2v.train_model()
        self.classifier.initialize_model()
        self.classifier.train_model(self.d2v, x_train, y_train, type_file, original_file)
        self.classifier.test_model(self.d2v, x_test, y_test, type_file, target_file)
        return self.d2v, self.classifier


def run(dataset_file, type_file, target_file, original_file):
    tc = TextClassifier()
    tc.read_data(dataset_file)
    tc.train_classifier(type_file, target_file, original_file)


if __name__ == "__main__":
    run("first_text.csv", "Text", "text_doc2vec_first.p", "train_first_embed.json")
    run("second_text.csv", "Text", "text_doc2vec_second.p", "train_second_embed.json")
    run("anomalies_text.csv", "Text", "text_doc2vec_anomalies.p", "train_anomalies_embed.json")
    run("first_triple.csv", "Triple", "triple_doc2vec_first.p", "train_first_embed.json")
    run("second_triple.csv", "Triple", "triple_doc2vec_second.p", "train_second_embed.json")
    run("anomalies_triple.csv", "Triple", "triple_doc2vec_anomalies.p", "train_anomalies_embed.json")

