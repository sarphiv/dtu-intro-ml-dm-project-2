from models.model import Model
import sklearn.linear_model as lm


class LinearRegression(Model):
    def train_predict(self, train_features, train_labels, test_features):
        m = lm.LinearRegression(fit_intercept=True).fit(train_features, train_labels)
        return m.predict(test_features)