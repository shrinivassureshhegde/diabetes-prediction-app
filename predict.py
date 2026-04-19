import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def predict_diabetes(data):
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)
    return model.predict(data)[0]


if __name__ == "__main__":
    sample = [2, 120, 70, 20, 80, 25.0, 0.5, 30]
    print(predict_diabetes(sample))