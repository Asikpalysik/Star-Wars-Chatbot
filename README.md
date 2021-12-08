# Star-Wars-Chatbot

Simple chatbot implementation with PyTorch.
A chatbot made in Python that features various data about the Star Wars universe.
This is a generic chatbot. Can be trained on pretty much any conversation as long as formatted correctly JSON file. I used it for a final project in Artificial Intelligence. To use just run the script training first, then run your chatbot. 

## Introduction

Chatbots are extremely helpful for business organizations and also the customers. The majority of people prefer to talk directly from a chatbox instead of calling service centers. Today I am going to build an exciting project on Chatbot. I will implement a chatbot from scratch that will be able to understand what the user is talking about and give an appropriate response. Chatbots are nothing but an intelligent piece of software that can interact and communicate with people just like humans. Here in this project we created an AI Chatbot which is focused for The Star Wars Cinematic Universe and trying training it in such a way that it can answer some of the basics queries about Star Wars.

## Explanation Of Chatbot

Chatbots are basically AI intelligence bots which can interact with the user or customers depends upon the usage. It is an application of Artificial Intelligence and Machine Learning¬. Now-a-days technology is increasing rapidly. In this technological world every industry is trying to automate things to provide better services. One of the great application of automation would be chatbot. 

## There are basically two types of Chatbots :

-Command based: Chatbots that function on predefined rules and can answer to only limited queries or questions. Users need to select an option to determine their next step.
-Intelligent/AI Chatbots: Chatbots that leverage Machine Learning and Natural Language Understanding to understand the user’s language and are intelligent enough to learn from conversations with their users. You can converse via text, speech or even interact with a chatbot using graphical interfaces.

All chatbots come under the NLP (Natural Language Processing) concepts. NLP is composed of two things:
•	NLU (Natural Language Understanding): The ability of machines to understand human language like English.
•	NLG (Natural Language Generation): The ability of a machine to generate text similar to human written sentences
Imagine a user asking a question to a chatbot: “Hey, what’s on the news today?” The chatbot will break down the user sentence into two things: intent and an entity. The intent for this sentence could be get_news as it refers to an action the user wants to perform. The entity tells specific details about the intent, so "today" will be the entity. So this way, a machine learning model is used to recognize the intents and entities of the chat.

## Strategy

•	Import Libraries and Load the Data
•	Preprocessing the Data
•	Create Training and Testing Data
•	Training the Model
•	Graphical user interface

## Import Libraries and Load the Data

I created a new python file and name it as chatbot.py and then import all the required modules. After that I loaded starwarsintents.json data file in our Python program.

```
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
import torch
import torch.nn as nn
import random
import json
from torch.utils.data import Dataset, DataLoader
from tkinter import *

with open("starwarsintents.json", "r") as f:
    intents = json.load(f)
 ```

## Preprocessing the Data

•	Creating Custom Functions:

We will create custom Functions so that it is easy for us to implement afterwards. Natural language (nltk) took kit is a really useful library that contains important classes that will be useful in any of your NLP task. To know a bit more about Natural language (nltk). Please click here for more information.

•	Stemming:

If we have 3 words like “walk”, “walked”, “walking”, these might seem different words but they generally have the same meaning and also have the same base form; “walk”. So, in order for our model to understand all different form of the same words we need to train our model with that form. This is called Stemming. There are different methods that we can use for stemming. Here we will use Porter Stemmer model form our NLTK Library. For more information click here.






•	Bag of Words:

We will be splitting each word in the sentences and adding it to an array. We will be using bag of words. Which will initially be a list of zeros with the size equal to the length of the all words array.If we have a array of sentences = ["hello", "how", "are", "you"] and an array of total words = ["hi", "hello", "I", "you", "bye", "thank", "cool"] then its bag of words array will be bog = [ 0 , 1 , 0 , 1 , 0 , 0 , 0].We will loop over the each word in the all words array and the bog array corresponding to each word. If a word from the sentence is found in the all words array, 1 will be replaced at that index/position in bag array. Click here for more information.
During the the process , we will also use nltk.word_tokenize() which will convert a single sentence string into a list of word. E.g if you pass "hello how are you", it will return ["hello", "how", "are", "you"].
Note: we will pass lower case words to the stemmer so that words like Good and good (capitalized) won’t be labelled as different words.
```
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
```

In order to get the right information, we will be unpacking starwarsintents.json it with the following code.

```
all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents["intents"]:
    tag = intent["tag"]
    # add to tag list
    tags.append(tag)
    for pattern in intent["patterns"]:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))
print(xy)
```

This will separate all the tags & words into their separate lists

```
# stem and lower each word
ignore_words = ["?", ".", "!"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)
```

Create Training and Testing Data

We will transform the data into a format that our PyTorch Model can easily understand. One hot encoding Is the process of splitting multiclass or multi valued data column to separate columns and labelling the cell 1 in the row where it exists. (we won’t use it so don’t worry about it). Click here to know more about CrossEntopyLoss.

```
# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train)
print(y_train)
```

•	PyTorch Model
	
Here we will be making a class to implement our custom Neural Network. It will be a Feed Forward Neural Network which will have 3 Linear Layers and we will be using activation function “ReLU”. For more click here.

•	Feed Forward Neural Network

A feedforward neural network is an artificial neural network wherein connections between the nodes do not form a cycle. As such, it is different from its descendant: recurrent neural networks.The feedforward neural network was the first and simplest type of artificial neural network devised In this network, the information moves in only one direction—forward—from the input nodes, through the hidden nodes (if any) and to the output nodes. There are no cycles or loops in the network. Click here to know more.

•	Activation Function

An activation function is a function used in artificial neural networks which outputs a small value for small inputs, and a larger value if its inputs exceed a threshold. If the inputs are large enough, the activation function "fires", otherwise it does nothing. In other words, an activation function is like a gate that checks that an incoming value is greater than a critical number. 
Activation functions are useful because they add non-linearities into neural networks, allowing the neural networks to learn powerful operations. If the activation functions were to be removed from a feedforward neural network, the entire network could be re-factored to a simple linear operation or matrix transformation on its input, and it would no longer be capable of performing complex tasks such as image recognition. Some more information here.

•	ReLU Function:

There are a number of widely used activation functions in deep learning today. One of the simplest is the rectified linear unit, or ReLU function which is a piece wise linear function that outputs zero if its input is negative, and directly outputs the input otherwise:
Mathematical definition of the ReLU Function
Graph of the ReLU function, showing its flat gradient for negative x. For more click here.






•	ReLU Function Derivative:

It is also instructive to calculate the gradient of the ReLU function, which is mathematically undefined at x = 0 but which is still extremely useful in neural networks.The derivative of the ReLU function. In practice the derivative at x = 0 can be set to either 0 or 1. The zero derivative for negative x can give rise to problems when training a neural network, since a neuron can become 'trapped' in the zero region and backpropagation will never change its weights.



Creating our model. Here we have inherited a class from NN.Module because we will be customizing the model & its layers

```
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
```

We will use some Magic functions, write our class. You can read online about __getitem__   and  __setitem__  magic funtions. 

```
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
```

Every Neural network has a set of hyper parameters that need to be set before use.
Before Instantiating our Neural Net Class or Model that we wrote earlier, we will first define some hyper parameters which can be changed accordingly. Click here for more.

```
# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)
```

We will now instantiate the model, loss and optimizer functions.

•	Loss Function: Cross Entropy here
•	Optimizer: Adam Optimizer here

```
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

Training Model

```
# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}
```

Saving training model.

```
FILE = "data.pth"
torch.save(data, FILE)
```

Loading Data

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("starwarsintents.json", "r") as json_data:
    intents = json.load(json_data)
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
```

Our Model is Ready. As our training data was very limited, we can only chat about a handful of topics. You can train it on a bigger dataset to increase the chatbot’s generalization / knowledge.

```
bot_name = "BARDROID9"
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    return "Sorry, didn't get it..."
```

Graphic User Interface 

Here you can find beautiful interface created by Patrik Loeber . 
On same I will advise you to have a look at his tutorials here and GitHub repository.

```
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"
class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
    def run(self):
        self.window.mainloop()
    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)
        # hea label
        head_label = Label(
            self.window,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            text="Welcome",
            font=FONT_BOLD,
            pady=10,)
        head_label.place(relwidth=1)
        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        # text widget
        self.text_widget = Text(
            self.window,
            width=20,
            height=2,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            font=FONT,
            padx=5,
            pady=5,)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)
        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        # send button
        send_button = Button(
            bottom_label,
            text="Send",
            font=FONT_BOLD,
            width=20,
            bg=BG_GRAY,
            command=lambda: self._on_enter_pressed(None),)
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
    def _insert_message(self, msg, sender):
        if not msg:
            return
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)
if __name__ == "__main__":
    app = ChatApplication()
    app.run()
```

Our Model has been trained with very few examples, so it does not understand everything. It is a great start for new learners. Just train this exact model on a bigger dataset & Voila, You'll see the charm .





Questions for Chatbot as an Example


Hello!
What can you do?
Are you alive?
Who am I?
Who is your creator?
Tell me about Mr. ASLAN?
Which items do you have in your bar?
I am looking for help.
I need assistance in my mission?
Tell me top 10 jedi?
Tell me top 10 sith?
Tell me top 10 bounti hounter?
Who is the best jedi in Galaxy?
Tell me top 10 sith?
I am looking for the best bounti hounter in galaxi?
Tell me a joke!
Tell me a story?
See you later?
abra cadabra












Summery

So now we have a chatbot framework, a recipe for making it a stateful service, and a starting-point for adding context. Most chatbot frameworks in the future will treat context seamlessly. Think of creative ways for intents to impact and react to different context settings. Your users’ context dictionary can contain a wide-variety of conversation context.

Files

starwarsintents.json
Chatbot.py
Chatbot.ipynb
data.pth

References

https://machinelearningmastery.com/natural-language-processing/
http://snowball.tartarus.org/algorithms/porter/stemmer.html
https://machinelearningmastery.com/gentle-introduction-bag-words-model/
https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
https://brilliant.org/wiki/feedforward-neural-networks/
https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
https://newbedev.com/understanding-getitem-method
https://www.codespeedy.com/__setitem__-and-__getitem__-in-python-with-example/
https://towardsdatascience.com/understanding-hyperparameters-and-its-optimisation-techniques-f0debba07568
https://machinelearningmastery.com/cross-entropy-for-machine-learning/
https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077
https://github.com/python-engineer
https://www.youtube.com/watch?v=RpWeNzfSUHw&list=PLqnslRFeH2UrFW4AUgn-eY37qOAWQpJyg
