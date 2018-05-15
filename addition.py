from keras.models import Sequential
from keras.layers import Dense
import numpy
seed = 7
numpy.random.seed(seed)

class GenderVerifier(object):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(6, input_dim=5, init='uniform', activation='relu'))
        self.model.add(Dense(3, init='uniform', activation='relu'))
        self.model.add(Dense(1, init='uniform', activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

    def train(self, csv_filepath: str):
        dataset = numpy.loadtxt(csv_filepath, delimiter=",")
        X = dataset[:, 0:5]
        Y = dataset[:, 5]
        self.model.fit(X, Y, epochs=270, batch_size=5, verbose=2)

    def predict_by_csv(self, csv_filepath: str) -> list:
        dataset = numpy.loadtxt(csv_filepath, delimiter=",")
        X = dataset[:, 0:5]
        Y = dataset[:, 5]
        predictions = self.model.predict(X)
        return [round(x[0]) for x in predictions]

    def predict_by_input(self):
        hair_length = int(input('Введите длину волос (0 - короткие, 1 - длинные'))
        shoes = int(input('Введите обувь (0 - кеды, 1 - туфли на каблуках, 2 - оксфорды)'))
        legs = int(input('Введите одежду на ногах (0 - шорты, 1 - юбка, 2 - брюки, 3 - джинсы)'))
        body = int(input('Введите одежду на торс (0 - рубашка, 1 - майка, 2 - блузка, 3 - футболка)'))
        jewelry = int(input('Введите украшения (0 - нет, 1 - кольцо, 2 - ожерелье, 3 - все вместе)'))
        x = numpy.array([[hair_length, shoes, legs, body, jewelry]])
        prediction = self.model.predict(x)
        return  [round(x[0]) for x in prediction]


if __name__ == '__main__':
    verifier = GenderVerifier()
    verifier.train('traindata.csv')
    while True:
        print(verifier.predict_by_input())