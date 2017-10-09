import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import f1_score

data_frame = pd.read_csv("dataset/train.csv", usecols=["Survived", "Age", "Sex"])

new_df = data_frame.dropna()
def encode_sex(row):
    if row["Sex"] == "male":
        return 1
    return 0
def encode_survived(x):
    return [0,1] if x == 0 else [1,0]
def one_hot_encode_sex(row):
    sex = [1,0] if row[1] == 1 else [0,1]
    new_row = [0,0,0]
    new_row[0] = row[0]
    new_row[1] = sex[0]
    new_row[2] = sex[1]
    return new_row;

new_df["sex"] = new_df.apply(lambda row : encode_sex(row), axis=1)
df = new_df.drop(["Sex"], axis=1)


df = shuffle(df)
df_test = df[0:len(df)/10]
df_train = df[len(df)/10:len(df)]

x_train = df_train[["sex", "Age"]].values.tolist()
x_test = df_test[["sex", "Age"]].values.tolist()

y_train = df_train["Survived"].values.tolist()
y_test = df_test["Survived"].values.tolist()

x_test = list(map(one_hot_encode_sex, x_test))
x_train = list(map(one_hot_encode_sex, x_train))

y_test = list(map(encode_survived, y_test))
y_train = list(map(encode_survived, y_train))
features = 3
classes = 2
hidden_units = 5

# print y_test
# print len(new_df)
# print len(data_frame)

X = tf.placeholder(tf.float32, shape=[None, features], name='X')
W1 = tf.Variable(tf.random_normal([features, hidden_units]))
B1 = tf.Variable(tf.random_normal([hidden_units]))
A = tf.nn.softmax(tf.matmul(X, W1) + B1)

W2 = tf.Variable(tf.random_normal([hidden_units, classes]))
B2 = tf.Variable(tf.random_normal([classes]))
Y_ = tf.nn.softmax(tf.matmul(A, W2) + B2)


Y = tf.placeholder(tf.float32, shape=[None, classes], name='Y')

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(Y_), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
epochs = 2000
for i in range(epochs):
    sess.run(train_step, feed_dict={X: x_train, Y: y_train})
    #print "{} / {} done".format(i, epochs)
# print sess.run(Y_)
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
