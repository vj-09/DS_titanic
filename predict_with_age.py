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

y_test = np.transpose(list(map(encode_survived, y_test)))
y_train = np.transpose(list(map(encode_survived, y_train)))
features = 3
classes = 2
hidden_units = 5

# print y_test
# print len(new_df)
# print len(data_frame)

X = tf.placeholder(tf.float32, shape=[None, features], name='X')
# Hidden layer 1
W1 = tf.Variable(tf.random_normal([hidden_units, features]), name='W1')
A1 = tf.sigmoid(tf.matmul(W1, X, transpose_b=True), name='A1')
# Output layer
W2 = tf.Variable(tf.random_normal([classes, hidden_units]), name='W2')
H = tf.transpose(tf.nn.softmax(tf.transpose(tf.matmul(W2, A1))), name='H')
# Cost function
Y = tf.placeholder(tf.float32, shape=[classes, None], name='Y')
j = -tf.reduce_sum(Y * tf.log(H), name='j')

optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(j)

session = tf.Session()
session.run(tf.initialize_all_variables())
epochs = 2500
cost = []
iteration = []
for epoch in range(epochs):
    cost.append(session.run(j, feed_dict={X: x_train, Y: y_train}))
    iteration.append(epoch)
    session.run(optimizer, feed_dict={X: x_train, Y: y_train})


y_pred = np.transpose(session.run(H, feed_dict={X: x_test})).argmax(1)
y_true = np.transpose(y_test).argmax(1)
print y_pred
f1 = f1_score(y_true, y_pred, average=None)
print('================================================================')
print('True: {}'.format(y_true))
print('Pred: {}'.format(y_pred))
print 'F1 score: {!r}'.format(f1)
print('================================================================')
# hidden_out = tf.add(tf.matmul(x, W1), b1)
# hidden_out = tf.nn.relu(hidden_out)
#
# y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
#
# squared_delta = tf.square(y_ - y)
# loss = tf.reduce_sum(squared_delta)
#
# optimizer = tf.train.GradientDescentOptimizer(0.001)
# train = optimizer.minimize(loss)

# init_op = tf.global_variables_initializer()
