import cv2

dataset = cv2.CascadeClassifier('data.xml')

capture = cv2.VideoCapture(0)
facedata = []
while True:
    ret,img = capture.read()
    # print(ret)
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),4)

            face = gray[y:y+h, x:x+w]
    #        face = cv2.resize(face, (64,64))
            if len(facedata) < 2:
                facedata.append(face)
#                print(len(facedata))

            cv2.imshow('result', img)
            cv2.imwrite('image1.jpg', face)
        if cv2.waitKey(1) == 27 or len(facedata) >= 2:
            break
    else:
        print("Camera not working")

cv2.destroyAllWindows()
capture.release()

from keras.models import load_model

# load model
model = load_model('model.h5')
model2 = load_model('model_100.h5')

import warnings
warnings.filterwarnings("ignore")
import ftfy
import nltk
import numpy as np
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image

# prediction from model1
test_image = image.load_img('image1.jpg', target_size = (28,28))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

predicted_class_indices=np.argmax(result,axis=1)

labels = {'depressed': 0, 'not depressed': 1}

labels = dict((v,k) for k,v in labels.items())
prediction_ = [labels[k] for k in predicted_class_indices]
prediction_model = prediction_[0]

# prediction from model2
# Expand Contraction
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = str(tweet)
        # if url links then dont append to avoid news articles
        # also check tweet length, save those > 10 (length of word "depression")
        if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
            #remove hashtag, @mention, emoji and image URLs
            tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", tweet).split())
            
            #fix weirdly encoded texts
            tweet = ftfy.fix_text(tweet)
            
            #expand contraction
            tweet = expandContractions(tweet)

            #remove punctuation
            tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

            #stop words
            stop_words = set(stopwords.words('english'))
            word_tokens = nltk.word_tokenize(tweet) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            tweet = ' '.join(filtered_sentence)

            #stemming words
#            tweet = PorterStemmer().stem(tweet)
            tweet = WordNetLemmatizer().lemmatize(tweet)
            
            cleaned_tweets.append(tweet)

    return cleaned_tweets

tokenizer = Tokenizer(num_words=20000)

def example(string):
    clean_string = clean_tweets(string)
    sequence = tokenizer.texts_to_sequences(clean_string)
    data = pad_sequences(sequence, maxlen=140)
    pred = model2.predict(data)
    pred = np.average(pred)
    pred = np.round(pred)
    if pred==0:
        return ("depressed")
    else:
        return("not depressed")
        
#strings = ["Trouble concentrating, remembering details, and making decisions","Fatigue",
#           "Feelings of guilt, worthlessness, and helplessness","Pessimism and hopelessness",
#           "Insomnia, early-morning wakefulness, or sleeping too much","Irritability",
#           "Restlessness","Loss of interest in things once pleasurable, including sex",
#           "Overeating, or appetite loss","Aches, pains, headaches, or cramps that won't go away",
#           "Digestive problems that don't get better, even with treatment",
#           "Persistent sad, anxious, or empty feelings","Suicidal thoughts or attempts",
#           "kill"]

strings = ["Just the labour involved in creating the layered richness of the imagery in this chiaroscuro of madness and light is astonishing."]
#prediction_model2 = []
#for i in range(len(strings)):
#    a = example(strings[i])
#    prediction_model2.append(a)

prediction_model2 = example(strings)
    
print(prediction_model,prediction_model2)
#print(prediction_model2)

# final output
if prediction_model==prediction_model2:
    if prediction_model=="depressed":
        output = "Major Depression"
    else:
        output = "No depression"
else:
    output = "Minor depression"
    
print(output)
