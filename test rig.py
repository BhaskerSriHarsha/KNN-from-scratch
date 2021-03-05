import keras
from keras.datasets import fashion_mnist
from knn import KNN

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

x_train = x_train/255
x_test = x_test/255

print(x_train.shape)
print(x_test.shape)

clf = KNN(k=3,loss="test")
clf.train(x_train,y_train)
predictions = clf.evaluate(x_test[:50],y_test[:50])

print(predictions)

