import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

class Validator():

    def __init__(self, D, labels):
        self.D = D
        self.labels = labels

    def train_linear(self):

        correct = 0
        total = 0
        results = []

        for left_out in range(self.D.shape[0]):
            test_vec = self.DD[left_out]
            test_label = self.labels[left_out]

            D_train = self.D[[element for i, element in enumerate(self.D) if i != left_out]]
            labels_train = self.labels[[element for i, element in enumerate(self.labels) if i != left_out]]
            clf_svm = LinearSVC()
            clf_svm.fit(D_train, labels_train)
            y_pred = clf_svm.predict(tes_vec)
            results.append(y_pred)
            if y_pred == test_label:
                correct += 1
            total += 1
        print 'percent accuracy: ', correct/total
        return results
