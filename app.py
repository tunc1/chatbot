import json
import numpy
import random
from nltk.stem import WordNetLemmatizer
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,Dense,LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

x=[]
y=[]
stopwords=nltk.corpus.stopwords.words("english")
lemmatizer=WordNetLemmatizer()
tokenizer=Tokenizer(oov_token="UNK")
maxlength=10
encoder=LabelEncoder()

with open("data.json") as file:
    data=json.load(file)
for each in data:
    for pattern in each["patterns"]:
        x.append(pattern.lower())
        y.append(each["class"])

y=encoder.fit_transform(y)
y=to_categorical(y)
labels=encoder.classes_

tokenizer.fit_on_texts(x)
vocabulary_size=len(tokenizer.word_index)+1

array=[]
for text in x:
    tokens=nltk.word_tokenize(text)
    tmp=[word for word in tokens if word not in stopwords]
    tmp=[lemmatizer.lemmatize(word) for word in tmp]
    array.append(" ".join(tmp))
x=array

def process_texts(texts):
    texts=tokenizer.texts_to_sequences(texts)
    return pad_sequences(texts,maxlength,truncating="post")

x=process_texts(x)

model=Sequential([
    Embedding(input_dim=vocabulary_size,output_dim=32,input_length=maxlength),
    LSTM(8),
    Dense(len(labels),activation="softmax")
])
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics="accuracy")
model.fit(x,y,batch_size=1,epochs=100)

test=process_texts(["Hi"])
prediction=numpy.argmax(model.predict(test))
responses=data[prediction]["responses"]
random_index=random.randint(0,len(responses)-1)
print(responses[random_index])