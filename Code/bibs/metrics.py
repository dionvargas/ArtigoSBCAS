import math
import numpy as np

class CategoricalMetrics():
    __metrics = {}
    __true_positives = None
    __true_negatives = None
    __false_positives = None
    __false_negatives = None
    __accuracy = None
    __sensitivity_micro = None
    __specificity_micro = None
    __precision_micro = None
    __f1_micro = None
    __g_mean_micro = None
    __sensitivity_macro = None
    __specificity_macro = None
    __precision_macro = None
    __f1_macro = None
    __g_mean_macro = None

    def __init__(self, cm, label=''):
        self.cm = cm
        self.label = label

    def get_true_positives(self):
        if self.__true_positives is None:
            self.__true_positives = np.sum(np.diag(self.cm))

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'true_positives' + self.label: self.__true_positives})

        return self.__true_positives

    def get_true_negatives(self):
        if self.__true_negatives is None:
            suport = []
            for i in range(len(self.cm)):
                for j, c in enumerate(self.cm):
                    if i != j:
                        suport.append(sum(c) - c[i])

            self.__true_negatives = np.sum(suport)

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'true_negatives' + self.label: self.__true_negatives})

        return self.__true_negatives

    def get_false_positives(self):
        if self.__false_positives is None:
            temp = []
            for i in range(len(self.cm)):
                temp.append(sum(self.cm[:, i]) - self.cm[i, i])
            self.__false_positives = np.sum(temp)

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'false_positives' + self.label: self.__false_positives})

        return self.__false_positives

    def get_false_negatives(self):
        if self.__false_negatives is None:
            temp = []
            for i in range(len(self.cm)):
                temp.append(sum(self.cm[:, i]) - self.cm[i, i])
            self.__false_negatives = np.sum(temp)

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'false_negatives' + self.label: self.__false_negatives})

        return self.__false_positives

    def get_accuracy(self):
        if self.__accuracy is None:
            self.__accuracy = np.sum(self.get_true_positives()) / np.sum(self.cm)

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'accuracy' + self.label:self.__accuracy})

        return self.__accuracy

    def get_sensitivity_micro(self):
        if self.__sensitivity_micro is None:
            self.__sensitivity_micro = self.get_accuracy()

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'sensitivity_micro' + self.label:self.__sensitivity_micro})

        return self.__sensitivity_micro

    def get_specificity_micro(self):
        if self.__specificity_micro is None:
            self.__specificity_micro = self.get_accuracy()

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'specificity_micro' + self.label:self.__specificity_micro})

        return self.__specificity_micro

    def get_precision_micro(self):
        if self.__precision_micro is None:
            self.__precision_micro = self.get_accuracy()

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'precision_micro' + self.label:self.__precision_micro})

        return self.__precision_micro

    def get_f1_micro(self):
        if self.__f1_micro is None:
            self.__f1_micro = self.get_accuracy()

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'f1_micro' + self.label:self.__f1_micro})

        return self.__f1_micro

    def get_g_mean_micro(self):
        if self.__g_mean_micro is None:
            self.__g_mean_micro = self.get_accuracy()

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'g_mean_micro' + self.label:self.__g_mean_micro})

        return self.__g_mean_micro

    def get_precision_macro(self):
        if self.__precision_macro is None:

            suport = []
            for i, c in enumerate(self.cm):
                suport.append(c[i] / sum(c))

            self.__precision_macro = np.average(suport)

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'precision_macro' + self.label:self.__precision_macro})

        return self.__precision_macro

    def get_sensitivity_macro(self):
        if self.__sensitivity_macro is None:

            suport = np.zeros(len(self.cm))
            for c in self.cm:
                for i, e in enumerate(c):
                    suport[i] += c[i]
            for i, c in enumerate(self.cm):
                suport[i] = c[i] / suport[i]
            self.__sensitivity_macro = np.average(suport)

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'sensitivity_macro' + self.label:self.__sensitivity_macro})

        return self.__sensitivity_macro

    def get_specificity_macro(self):
        if self.__specificity_macro is None:

            self.__specificity_macro = self.get_true_negatives()/(self.get_true_negatives() + self.get_false_negatives())

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'specificity_macro' + self.label:self.__specificity_macro})

        return self.__specificity_macro


    def get_f1_macro(self):
        if self.__f1_macro is None:
            self.__f1_macro = 2*(self.get_precision_macro()*self.__sensitivity_macro/(self.get_precision_macro()+self.__sensitivity_macro))

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'f1_macro' + self.label:self.__f1_macro})

        return self.__f1_macro

    def get_g_mean_macro(self):
        if self.__g_mean_macro is None:
            self.__g_mean_macro = math.sqrt(self.__sensitivity_macro*self.get_specificity_macro())

            # The feature is added into the "metrics" dictionary
            self.__metrics.update({'g_mean_macro' + self.label:self.__g_mean_macro})

        return self.__g_mean_macro

    def extract_all_metrics(self):
        self.get_true_positives()
        self.get_true_negatives()
        self.get_false_positives()
        self.get_false_negatives()
        self.get_accuracy()
        self.get_sensitivity_micro()
        self.get_specificity_micro()
        self.get_precision_micro()
        self.get_f1_micro()
        self.get_g_mean_micro()
        self.get_sensitivity_macro()
        self.get_specificity_macro()
        self.get_precision_macro()
        self.get_f1_macro()
        self.get_g_mean_macro()

    def allMetrics(self):
        return self.__metrics