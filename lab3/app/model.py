import numpy as np
from sklearn.linear_model import LinearRegression

class Model:
    def __init__(self):
        self.model = LinearRegression()
        # обучение модели
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        self.model.fit(X, y)

    def predict(self, x):
        x = np.array(x).reshape(-1, 1)
        prediction = self.model.predict(x)
        return prediction.tolist()

#локальный тест(удобно для работы в группе)
if __name__ == "__main__":
    model = Model()
    print(model.predict([1, 2, 3]))