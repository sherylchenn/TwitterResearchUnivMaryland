import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Input
import keras_nlp
from sklearn.utils import shuffle
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
from datetime import datetime


now = datetime.now()

DATASET_ENCODING = "ISO-8859-1"

training = False

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

#kaggle datasets download -d kazanova/sentiment140
dfx = pd.read_csv("API_v2/02_09:54:46.csv", escapechar='\\', encoding=DATASET_ENCODING)

#dfx = pd.read_csv("/Users/sherylchen/PycharmProjects/pythonProject/tweepy/examples/API_v2/02_09\:05\:19.csv.zip ",encoding=DATASET_ENCODING)

#df = pd.read_csv("/Users/sherylchen/Downloads/training.1600000.processed.noemoticon.csv.zip",encoding=DATASET_ENCODING)
df = pd.read_csv("/Users/sherylchen/Downloads/archive.zip",encoding=DATASET_ENCODING)
print("Cread_csv done")

df= df.iloc[:,[0,-1]]
df.columns = ['sentiment','tweet']
df = pd.concat([df.query("sentiment==0").sample(750000),df.query("sentiment==4").sample(750000)])
df.sentiment = df.sentiment.map({0:0,4:1})
df =  shuffle(df).reset_index(drop=True)

df,df_test = train_test_split(df,test_size=0.2)

def vectorize(df):
    embeded_tweets = embed(df['tweet'].values.tolist()).numpy()
    targets = df.sentiment.values
    return embeded_tweets,targets

hdf = df.head(5)
print("loading_embed")

# universal-sentence-encoder is pretrained transformer
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("loading done <3")

s = embed(['hi samuels, this is our project']).numpy().shape



embeded_tweets,targets = vectorize(df)

model_output = '/Users/sherylchen/Downloads/transformer1116'
file_exists = os.path.exists(model_output)
if file_exists:
    print("loading existing model")
    model = keras.models.load_model(model_output)



from keras.layers import Flatten, LSTM

# add three more layers on top of transformer based universal-sentence-encoder  to fine train for sentiment analysis purpose

model = Sequential()

#input layer take 512-dim vector from univeral sententence encoder
model.add(Input(shape=(512,),dtype='float32'))


# I trained this model with a public tweets dataset with 1.6M polarity labels

# First MLP (Multilayer perceptron ) layer decode the input to a 128-dim vector
model.add(Dense(128, activation = 'relu'))
# Second MLP   layer decode the input to a 64-dim vector
model.add(Dense(64, activation = 'relu'))
# Third MLP   layer decode the input to a 2-dim vector
model.add(Dense(2, activation='softmax'))
# Final softmax layer to get final prediction of polarity
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.summary()

num_epochs = 10
batch_size = 32   ## 2^x
from keras.utils import np_utils

if training:
    print("training")
    new_targets = np_utils.to_categorical(targets)
    history = model.fit(embeded_tweets,
                        new_targets,
                        epochs=num_epochs,
                        validation_split=0.1,
                        shuffle=True,
                        batch_size=batch_size)
    print("training_done")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


from sklearn.metrics import accuracy_score

dfx= dfx.iloc[:,[0,-1]]
dfx.columns = ['sentiment','tweet']
dfx = pd.concat([dfx.query("sentiment==0")])
#df.sentiment = df.sentiment.map({9:0,4:1})
#dfx =  shuffle(dfx).reset_index(drop=True)
embed_x,targets = vectorize(dfx)
print("predicting")


predictions = model.predict(embed_x)

predictions=np.argmax(predictions, axis=1)


def listToString(s):
    # initialize an empty string
    string = " "

    # return string
    return (string.join(s))


# Driver code
s = predictions

filename = '02_{}.csv'.format(current_time)
with open(filename, 'w') as f:


    print('filename:'+ filename)

    output = values = ",".join(map(str, predictions))
    f.write(output)
## output predictions and tweets to another output text file

#accuracy_score(predictions,targets)*100

#custom model
embed_test,targets = vectorize(df_test)
predictions = model.predict(embed_test)
predictions=np.argmax(predictions, axis=1)
accuracy = accuracy_score(targets, predictions)*100
print ('accuracy:{}'.format(accuracy))
model.save(model_output)





