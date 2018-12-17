import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def readData():
    # Read the training data and separate the dependent variable from the features
    trainDataFrame = pd.read_csv("final_processed_3.csv")
    
    # df.columns is zero-based pd.Index
    # X = trainDataFrame.drop(['div','FTR', 'key', 'date', 'WHH','WHD','WHA','home_player_1','home_player_2','home_player_3','home_player_4','home_player_5','home_player_6','home_player_7','home_player_8','home_player_9','home_player_10','home_player_11','away_player_1','away_player_2','away_player_3','away_player_4','away_player_5','away_player_6','away_player_7','away_player_8','away_player_9','home_team_overall','away_team_overall'], axis=1)
    # X = trainDataFrame.drop(['div','FTR', 'key', 'date', 'WHH','WHD','WHA', 'home_team_overall','away_team_overall', 'Bsay n365H','B365D','B365A','BWH','BWD','BWA','GBH','GBD','GBA','IWH','IWD','IWA','LBH','LBD','LBA','SBH','SBD','SBA','WHH','WHD','WHA','SJH','SJD','SJA','VCH','VCD','VCA','BSH','BSD','BSA','Bb1X2','BbMxH','BbAvH','BbMxD','BbAvD','BbMxA','BbAvA','BbOU','BbMx_2.5','BbAv_2.5','BbMx_2.5','BbAv_2.5','BbAH','BbAHh','BbMxAHH','BbAvAHH','BbMxAHA','BbAvAHA','home_player_overall','away_player_overall','home_team_overall','away_team_overall'], axis=1)
    X = trainDataFrame.drop(['div','FTR', 'key', 'date', 'WHH','WHD','WHA', 'home_player_overall','away_player_overall','home_team_overall','away_team_overall'], axis=1)
    y = trainDataFrame['FTR']

    # print (y)
    
#     nCols = len(trainDataFrame.columns)
    
#     X = trainDataFrame[trainDataFrame.columns[0:nCols-1]].values
#     y = trainDataFrame[trainDataFrame.columns[nCols-1]].values
    
    # Encoding the dependent variable
    # Incase of labels such as "Yes" and "No", we want to map them to a corresponding label vector
    # that has all zeroes expect at the corresponding label.
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    
    return (X, Y)
    
def one_hot_encode(labels):
    nLabels = len(labels)
    nUniqueLabels = len(np.unique(labels))
    one_hot_encode = np.zeros((nLabels, nUniqueLabels))
    one_hot_encode[np.arange(nLabels), labels] = 1
    return one_hot_encode

X, Y = readData()
X, Y = shuffle(X, Y, random_state=1)

# print (Y)

# Dividing the training data into train and dev sets
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, )#random_state = 415)

# print (train_y)

# Inspect the shape after dividing into the right data sets
# print ((train_x.shape))
# print ((test_x.shape))
# print ((train_y.shape))
# print ((test_y.shape))

# Important parameters and modelling variables
learning_rate = 0.11
training_epochs = 1001
cost_history = np.empty(shape=[1], dtype=float)

# dimension of the feature vector
n_dim = X.shape[1]
print ("Number of features: ", n_dim)
# Number of possible labels / output classes
n_class = Y.shape[1]
print ("Number of labels: ", n_class)

# Define the number of hidden layers and Neurons in each layer
n_hidden_1 = 85
n_hidden_2 = 85
n_hidden_3 = 85
n_hidden_4 = 85

# input and output variables
x = tf.placeholder(tf.float32, [None, n_dim]) # 'None' tells that it can be any value
W = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class]) # 'None' tells that it can be any value

# Defining the model
def multiLayerPerceptron(x, weights, biases):
    
    layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
    layer_2 = tf.nn.relu(layer_2)
    # layer_2 = tf.nn.sigmoid(layer_2)

    # layer_3 = tf.add(tf.matmul(layer_2, weights["h3"]), biases["b3"])
    # layer_3 = tf.nn.relu(layer_3)
    # layer_3 = tf.nn.sigmoid(layer_3)

    # layer_4 = tf.add(tf.matmul(layer_3, weights["h4"]), biases["b4"])
    # layer_4 = tf.nn.relu(layer_4)
    
    layer_out = tf.matmul(layer_2, weights["out"]) + biases["out"]
    
    return layer_out

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    "h4": tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    "out": tf.Variable(tf.truncated_normal([n_hidden_3, n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    "b4": tf.Variable(tf.truncated_normal([n_hidden_4])),
    "out": tf.Variable(tf.truncated_normal([n_class]))
}


# Initialize all the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

y = multiLayerPerceptron(x, weights, biases)

costFunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
# trainingStep = tf.train.GradientDescentOptimizer(learning_rate).minimize(costFunction)
trainingStep =  tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(costFunction)

mseHistory = []
accHistory = []

someSession = tf.Session()
someSession.run(init)

for epoch in range(training_epochs):
    someSession.run(trainingStep, feed_dict={x: train_x, y_: train_y})
    cost = someSession.run(trainingStep, {x: train_x, y_: train_y})
    cost_history = np.append(cost_history, cost)
    correctPred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    pred_y = someSession.run(y, {x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = someSession.run(mse)
    mseHistory.append(mse_)
    acc = (someSession.run(acc, {x: train_x, y_: train_y}))
    accHistory.append(acc)

    if (epoch % 10 == 1):
        print ("\nepoch: ", epoch, " Acc: ", acc)
        
# save_path = saver.save(someSession, )
someSession.close()

print (mseHistory)
print (accHistory)