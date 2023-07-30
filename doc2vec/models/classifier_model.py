from .model import Model
from .doc2vec_model import doc2VecModel

import logging
import os
import inspect
import pickle
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
base_path = os.path.dirname(os.path.abspath(base_file_path))
project_dir_path = os.path.dirname(os.path.abspath(base_path))
classifiers_path = os.path.join(project_dir_path, 'classifiers')


class classifierModel(Model):
    def __init__(self):
        super().__init__()
        self.dict_two = dict()
        self.data_all = list()
        self.ids = list()
        self.one_len = 0

    def initialize_model(self):
        self.model = LogisticRegression()

    def train_model(self, d2v, training_vectors, training_labels, type_file, original_file):
        self.get_events_type(type_file, original_file)
        logging.info("Classifier training")
        train_vectors = doc2VecModel.get_vectors(
            d2v, len(training_vectors), 300, 'Train')
        self.one_len = len(train_vectors)

        # 保存嵌入向量
        for i in range(len(train_vectors)):
            if len(self.data_all[i]) > 20:
                self.dict_two[(str(self.ids[i]) + str(self.data_all[i][:20])+'...')] = train_vectors[i]
            else:
                self.dict_two[(str(self.ids[i]) + str(self.data_all[i]))] = train_vectors[i]

        self.model.fit(train_vectors, np.array(training_labels))
        training_predictions = self.model.predict(train_vectors)
        logging.info(
            'Training predicted classes: {}'.format(np.unique(
                training_predictions)))
        logging.info(
            'Training accuracy: {}'.format(
                accuracy_score(training_labels, training_predictions)))
        logging.info(
            'Training precision: {}'.format(
            precision_score(
            training_labels, training_predictions)))
        logging.info(
            'Training recall: {}'.format(
            recall_score(
            training_labels, training_predictions)))
        logging.info(
            'Training F1 score: {}'.format(
                f1_score(
                    training_labels, training_predictions,
                    average='weighted')))

    def test_model(self, d2v, testing_vectors, testing_labels, type_file, target_file):
        logging.info("Classifier testing")
        test_vectors = doc2VecModel.get_vectors(
            d2v, len(testing_vectors), 300, 'Test')

        # 保存嵌入向量
        for j in range(len(test_vectors)):
            if len(self.data_all[j]) > 20:
                u = (str(self.ids[self.one_len+j])+str(self.data_all[self.one_len+j][:20])+'...')
            else:
                u = (str(self.ids[self.one_len + j]) + str(self.data_all[self.one_len+j]))
            self.dict_two[u] = test_vectors[j]

        # 保存成 p 类型文件
        p_dir = os.path.join('Vectors', type_file)
        if not os.path.exists(p_dir):
            os.mkdir(p_dir)
        pic_path = os.path.join('Vectors', type_file, target_file)
        pic = open(pic_path, 'wb')
        pickle.dump(self.dict_two, pic)
        pic.close()

        testing_predictions = self.model.predict(test_vectors)
        logging.info(
            'Testing predicted classes: {}'.format(
                np.unique(testing_predictions)))
        logging.info(
            'Testing accuracy: {}'.format(
                accuracy_score(testing_labels, testing_predictions)))
        logging.info(
            'Testing precision: {}'.format(
            precision_score(
            testing_labels, testing_predictions)))
        logging.info(
            'Training recall: {}'.format(
            recall_score(
            testing_labels, testing_predictions)))
        logging.info(
            'Testing F1 score: {}'.format(
                f1_score(
                    testing_labels, testing_predictions,
                    average='weighted')))

    def predict(self, d2v, testing_vectors):
        logging.info("Classifier Predicting")
        test_vectors = doc2VecModel.get_vectors(
            d2v, len(testing_vectors), 300, 'Test')
        testing_predictions = self.model.predict(test_vectors)
        logging.info(testing_predictions)

    def get_events_type(self, type_file, original_file):
        original_path = os.path.join('Training_Data', original_file)
        with open(original_path, 'r', encoding='utf-8') as tse:
            for line in tse.readlines():
                item = json.loads(line)
                if type_file == 'Text':
                    if isinstance(item["text"], list):
                        item["text"] = ','.join(item["text"])
                    self.data_all.append(item["text"])
                elif type_file == 'Triple':
                    self.data_all.append(item["Triplet"])
                self.ids.append(item["Order_number"])
