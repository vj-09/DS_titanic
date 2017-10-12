import pandas as pd
import numpy as np
import tensorflow as tf
pd.options.mode.chained_assignment = None
dataset = pd.read_csv('dataset/train01.csv')

#dataset = dataset[['','Sex','Survived']].dropna()
train_data = dataset[['Sex','Family_class','Family_lenth']]
# print train_data.head(4)
train_target1 = dataset['Survived']
def encode_survived(x):
    return [0,1] if x == 0 else [1,0]
train_target = []
for a in range(len(train_target1)):
    # print train_target[a]
    train_target.append(encode_survived(train_target1[a]))
# print train_target
# print train_data['Sex']
train_data['Sex'] = pd.get_dummies(train_data['Sex']).values
# train_data['Sex_class'] = train_data['Sex'] * train_data['Pclass']
learning_rate = 0.007

training_epochs = 7500
display_step = 1
test_data = train_data.iloc[750:]
test_target = train_target[750:]
train_data =  train_data.iloc[:750]
train_target =  train_target[:750]
numF = train_data.shape[1]
numT = len(train_target[1])
X = tf.placeholder(tf.float32,[None,numF])
Y = tf.placeholder(tf.float32,[None,numT])
print numF , numT
train_data = train_data.values
# print train_data
# print train_target
w = tf.Variable(tf.zeros([numF,numT]))
b = tf.Variable(tf.zeros([numT]))
pred = tf.nn.softmax(tf.matmul(X,w)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred),reduction_indices=1))
print cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        print sess.run([w,b])
        _, c = sess.run([optimizer, cost], feed_dict={  X: train_data,
                                                          Y:train_target})
        print c
    print c
    correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={X: train_data, Y: train_target}))
#    print pred,cost
    print("Accuracy:", accuracy.eval({X:test_data, Y: test_target}))
