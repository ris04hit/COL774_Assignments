import matplotlib
matplotlib.use('TkAgg')

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import seaborn as sns
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from math import log10

def read_data(filename: str):   # function to read csv file using pandas
    df = pd.read_csv(filename)
    return df

def simplify_text(text: str):   # lower cases the text and replace all punctuation and numbers with space
    text = text.lower()
    punctuation = set(',.?/;:\'"\\|\{\}[]+=-_()*&^%$#@!~`<>1234567890’')
    text = ''.join(ch if ch not in punctuation else ' ' for ch in text)
    return text

def create_vocabulary(df: pd.DataFrame):      # function to create vocabulary from dataframe
    text = ' '.join(tweet for tweet in df[colName_text])
    word_set = set(text.split())
    vocab = {}
    index = 0
    for word in word_set:
        # vocab[index] = word
        vocab[word] = index
        index += 1
    return vocab

def nb_training(df: pd.DataFrame, vocabulary: dict):
    '''
        Trains a naive bayes model and returns classification parameters as well as conditional parameters
    '''
    vocab_size = len(vocabulary)
    num_class = len(classification)
    smoothing_parameter = 0.5
    cond_param = np.full((num_class, vocab_size), smoothing_parameter)           #phi_k|y, ones for laplace smoothing
    class_param = np.zeros((num_class,))            # phi_y
    class_word_count = np.full((num_class,), vocab_size*smoothing_parameter)             # Number of words in particular class
    for index, row in df.iterrows():
        class_val = classification[row[colName_class]]
        class_param[class_val]+=1
        for word in row[colName_text].split():
            class_word_count[class_val]+=1
            cond_param[class_val][vocabulary[word]]+=1
    class_param/=len(df)
    cond_param/=class_word_count[:, np.newaxis]
    return class_param, cond_param
    
def nb_classifier(cond_param_log: np.ndarray, class_param_log: np.ndarray, text: str, vocabulary: dict):        # classifies text into class
    class_belongs = 0
    log_prob = -float('inf')
    for class_val in range(class_param_log.shape[0]):     # Calculating log of probability for every class
        curr_log_prob = class_param_log[class_val]
        for word in text.split():
            if word in vocabulary:
                curr_log_prob += cond_param_log[class_val][vocabulary[word]]
        if curr_log_prob > log_prob:
            log_prob = curr_log_prob
            class_belongs = class_val
    return class_belongs

def model_accuracy(df: pd.DataFrame, classifier):     # returns model accuracy over set of data
    equality_series = df.apply(lambda row: classification[row[colName_class]] == classifier(row[colName_text]), axis = 1)
    correct_prediction = equality_series.sum()
    total_prediction = len(df)
    return correct_prediction/total_prediction

def create_wordcloud(df: pd.DataFrame, msg: str):     # creates word cloud
    for class_name, class_val in classification.items():
        classified_text = ' '.join(df.loc[df[colName_class]==class_name][colName_text])
        wc = WordCloud().generate(classified_text)
        fig = plt.figure()
        axes = fig.add_subplot()
        axes.imshow(wc, interpolation='bilinear')
        axes.set_title(f'{msg}: {class_name}')
        axes.axis('off')
        fig.savefig(f'{msg}: {class_name}')

def create_confusion_matrix(df: pd.DataFrame, classifier, msg: str):          # Creates confusion matrix
    true_labels = df[colName_class].apply(lambda class_name: classification[class_name])
    predicted_labels = df[colName_text].apply(classifier)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    fig = plt.figure(figsize=(8,6))
    axes = plt.subplot()
    labels = ['Negative', 'Positive', 'Neutral']
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax = axes, xticklabels=labels, yticklabels=labels)
    axes.set_xlabel('Predicted Labels')
    axes.set_ylabel('True Labels')
    axes.set_title(f'Confusion Matrix: {msg}')
    fig.savefig(f'Confusion Matrix: {msg}')

def stem_text(text):        # Returns the text after stemming
    txt = ''
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed_words = []
    for word in text.split():
        # stemmed_words.append(stemmer.stem(word))
        stemmed_words.append(lemmatizer.lemmatize(word))
    return ' '.join(word for word in stemmed_words)

def remove_stop_words(text):        # Removes stop word from text
    filtered_text = []
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union({'http', 'https', 'com', 'co', 'www' ,'t', 'amp', 'ac'})
    for word in text.split():
        if word not in stop_words:
            filtered_text.append(word)
    return ' '.join(word for word in filtered_text)

def nb_bigram_training(df: pd.DataFrame, vocabulary: dict):        # Trains naive bayes over bigrams
    vocab_size = len(vocabulary)
    num_class = len(classification)
    smoothening_factor = 0.5
    bigram_param = [{} for class_val in range(num_class)]
    class_word_count = np.full((num_class,), 0)             # Number of words in particular class
    for index, row in df.iterrows():
        text = row[colName_text].split()
        class_val = classification[row[colName_class]]
        class_word_count[class_val] += len(text) - 1
        for ind in range(len(text)-1):
            word1, word2 = text[ind], text[ind+1]       # Words for bigram
            for it_class_val in range(num_class):
                key = (vocabulary[word1], vocabulary[word2])
                if key not in bigram_param[it_class_val]:
                    bigram_param[it_class_val][key] = smoothening_factor      # Laplace smoothing
                    class_word_count[it_class_val] += smoothening_factor
                if it_class_val == class_val:
                    bigram_param[it_class_val][key] += 1
    for class_val in range(num_class):
        for key in bigram_param[class_val]:
            bigram_param[class_val][key]/=class_word_count[class_val]
            bigram_param[class_val][key] = log10(bigram_param[class_val][key])
    return bigram_param

def nb_bigram_classifier(cond_param_log: np.ndarray, class_param_log: np.ndarray, bigram_param_log: list, text: str, vocabulary: dict):    # Classifies text into class
    class_belongs = 0
    log_prob = -float('inf')
    bigram_contribution_factor = 0.6                        # To normalize bigram contribution feature
    for class_val in range(class_param_log.shape[0]):     # Calculating log of probability for every class
        curr_log_prob = class_param_log[class_val]
        splitted_text = text.split()
        for word in splitted_text:
            if word in vocabulary:
                curr_log_prob += cond_param_log[class_val][vocabulary[word]]
        for ind in range(len(splitted_text)-1):
            if (splitted_text[ind] in vocabulary) and (splitted_text[ind+1] in vocabulary):
                bigram = (vocabulary[splitted_text[ind]], vocabulary[splitted_text[ind+1]])
                if bigram in bigram_param_log[class_val]:
                    curr_log_prob += bigram_param_log[class_val][bigram]*bigram_contribution_factor
        if curr_log_prob > log_prob:
            log_prob = curr_log_prob
            class_belongs = class_val
    return class_belongs

def nb_length_training(df: pd.DataFrame, vocabulary: dict, interval_size: int):        # Trains naive bayes with length feature
    df_length = pd.DataFrame(df)
    df_length[colName_text] = df_length[colName_text].apply(lambda text: len(text.split()))
    max_length = df_length[colName_text].max()
    interval_num = max_length//interval_size + 1
    smoothening_factor = 10.0
    num_class = len(classification)
    length_param = np.full((num_class, interval_num), smoothening_factor)
    length_total_count = np.full((num_class,), interval_num*smoothening_factor)
    for index, row in df_length.iterrows():
        class_val = classification[row[colName_class]]
        length_total_count[class_val] += 1
        interval_val = row[colName_text]//interval_size
        length_param[class_val][interval_val] += 1
    length_param/=length_total_count[:, np.newaxis]
    return np.log10(length_param)

def nb_length_classifier(cond_param_log: np.ndarray, class_param_log: np.ndarray, length_param_log: np.ndarray, interval_size: int, text: str, vocabulary: dict):    # Classifies text into class
    class_belongs = 0
    log_prob = -float('inf')
    length_contribution_factor = 1.9                        # To normalize length contribution feature
    for class_val in range(class_param_log.shape[0]):     # Calculating log of probability for every class
        curr_log_prob = class_param_log[class_val]
        splitted_text = text.split()
        interval_val = len(splitted_text)//interval_size
        if interval_val < length_param_log.shape[1]:
            curr_log_prob += length_param_log[class_val][interval_val]*length_contribution_factor
        for word in splitted_text:
            if word in vocabulary:
                curr_log_prob += cond_param_log[class_val][vocabulary[word]]
        if curr_log_prob > log_prob:
            log_prob = curr_log_prob
            class_belongs = class_val
    return class_belongs

def nb_bigram_length_classifier(cond_param_log: np.ndarray, class_param_log: np.ndarray, bigram_param_log: list, length_param_log: list, interval_size: int, text: str, vocabulary: dict):    # Classifies text into class
    class_belongs = 0
    log_prob = -float('inf')
    bigram_contribution_factor = 0.6                        # To normalize bigram contribution feature
    length_contribution_factor = 1.9                        # To normalize length contribution feature
    for class_val in range(class_param_log.shape[0]):     # Calculating log of probability for every class
        curr_log_prob = class_param_log[class_val]
        splitted_text = text.split()
        interval_val = len(splitted_text)//interval_size
        if interval_val < length_param_log.shape[1]:
            curr_log_prob += length_param_log[class_val][interval_val]*length_contribution_factor
        for word in splitted_text:
            if word in vocabulary:
                curr_log_prob += cond_param_log[class_val][vocabulary[word]]
        for ind in range(len(splitted_text)-1):
            if (splitted_text[ind] in vocabulary) and (splitted_text[ind+1] in vocabulary):
                bigram = (vocabulary[splitted_text[ind]], vocabulary[splitted_text[ind+1]])
                if bigram in bigram_param_log[class_val]:
                    curr_log_prob += bigram_param_log[class_val][bigram]*bigram_contribution_factor
        if curr_log_prob > log_prob:
            log_prob = curr_log_prob
            class_belongs = class_val
    return class_belongs

def create_report(filename):        # create reports
    report_file = open(filename, 'w')
    report_file.write('Part(a)\n')
    report_file.write(f'Training Accuracy:\t\t\t{training_accuracy_raw}\n')
    report_file.write(f'Validation Accuracy:\t\t{validation_accuracy_raw}\n')
    report_file.write('\nPart(b)\n')
    report_file.write(f'Random Accuracy:\t\t\t{random_accuracy}\n')
    report_file.write(f'Positive Accuracy:\t\t\t{positive_accuracy}\n')
    report_file.write('\nPart(d)\n')
    report_file.write(f'Training Accuracy:\t\t\t{training_accuracy_processed}\n')
    report_file.write(f'Validation Accuracy:\t\t{validation_accuracy_processed}\n')
    report_file.write('\nPart(e)\n')
    report_file.write(f'Training Accuracy (Bigram):\t\t\t\t{training_accuracy_bigram}\n')
    report_file.write(f'Validation Accuracy (Bigram):\t\t\t{validation_accuracy_bigram}\n')
    report_file.write(f'Training Accuracy (length):\t\t\t\t{training_accuracy_length}\n')
    report_file.write(f'Validation Accuracy (length):\t\t\t{validation_accuracy_length}\n')
    report_file.write(f'Training Accuracy (Bigram+length):\t\t{training_accuracy_bigram_length}\n')
    report_file.write(f'Validation Accuracy (Bigram+length):\t{validation_accuracy_bigram_length}\n')
    report_file.write('\nPart(f(i))\n')
    for ind in range(len(split_perc)):
        report_file.write(f'Validation Accuracy ({split_perc[ind]}%):\t\t{validation_accuracy_da[ind]}\n')
    report_file.write('\nPart(f(ii))\n')
    for ind in range(len(split_perc)):
        report_file.write(f'Validation Accuracy ({split_perc[ind]}%):\t\t{validation_accuracy_split[ind]}\n')
    report_file.close()

# __Main__

train_filename = '../data/Corona_train.csv'
validation_filename = '../data/Corona_validation.csv'
colName_class = 'Sentiment'
colName_text = 'CoronaTweet'
classification = {'Negative': 0, 'Positive': 1, 'Neutral': 2}

# Part(a)
# Creating vocabulary
df_train = read_data(train_filename)
df_validate = read_data(validation_filename)
vocabulary = create_vocabulary(df_train)
print('Vocabulary Created for raw data')

# Training using Naive Bayes
class_param, cond_param = nb_training(df_train, vocabulary)
class_param_log = np.log10(class_param)
cond_param_log = np.log10(cond_param)
print('Trained Naive Bayes with raw data')

# Model Accuracy
training_accuracy_raw = model_accuracy(df_train, lambda text: nb_classifier(cond_param_log, class_param_log, text, vocabulary))
validation_accuracy_raw = model_accuracy(df_validate, lambda text: nb_classifier(cond_param_log, class_param_log, text, vocabulary))
print('Calculated Training Accuracy for raw data')

# # Word Cloud
create_wordcloud(df_train, "Raw Training Data")
create_wordcloud(df_validate, "Raw Validation Data")
print('Wordcloud for raw data created')
# plt.show()


# Part(b)
random_accuracy = model_accuracy(df_validate, lambda text: random.randint(0,2))
positive_accuracy = model_accuracy(df_validate, lambda text: 1)
print('Random and positive accuracy calculated')


# Part(c)
create_confusion_matrix(df_train, lambda text: nb_classifier(cond_param_log, class_param_log, text, vocabulary), 'Naive Bayes, Training data')
create_confusion_matrix(df_validate, lambda text: nb_classifier(cond_param_log, class_param_log, text, vocabulary), 'Naive Bayes, Validation data')
create_confusion_matrix(df_train, lambda text: random.randint(0,2), 'Random, Training Data')
create_confusion_matrix(df_validate, lambda text: random.randint(0,2), 'Random, Validation Data')
create_confusion_matrix(df_train, lambda text: 1, 'Positive, Training Data')
create_confusion_matrix(df_validate, lambda text: 1, 'Positive, Validation Data')
print('Confusion Matrix Created')
# plt.show()


# Part(d)
# Removing Punctuation, Case Sensitivity, Performing Stemming and Removing Stop Words
df_train_processed = pd.DataFrame(df_train)
df_validate_processed = pd.DataFrame(df_validate)
df_train_processed[colName_text] = df_train[colName_text].apply(simplify_text).apply(stem_text).apply(remove_stop_words)
df_validate_processed[colName_text] = df_validate[colName_text].apply(simplify_text).apply(stem_text).apply(remove_stop_words)

# Creating vocabulary
vocabulary = create_vocabulary(df_train_processed)
print('Vocabulary Created for processed data')

# Training using Naive Bayes
class_param, cond_param = nb_training(df_train_processed, vocabulary)
class_param_log = np.log10(class_param)
cond_param_log = np.log10(cond_param)
print('Trained Naive Bayes with processed data')

# Model Accuracy
training_accuracy_processed = model_accuracy(df_train_processed, lambda text: nb_classifier(cond_param_log, class_param_log, text, vocabulary))
validation_accuracy_processed = model_accuracy(df_validate_processed, lambda text: nb_classifier(cond_param_log, class_param_log, text, vocabulary))
print('Calculated Training Accuracy for processed data')

# Word Cloud
create_wordcloud(df_train_processed, "Processed Training Data")
create_wordcloud(df_validate_processed, "Processed Validation Data")
print('Wordcloud for processed data created')
# plt.show()


# Part(e)
# Training using Naive Bayes
bigram_param_log = nb_bigram_training(df_train_processed, vocabulary)
print("Trained Naive Bayes with bigram feature")

# Model Accuracy bigram
training_accuracy_bigram = model_accuracy(df_train_processed, lambda text: nb_bigram_classifier(cond_param_log, class_param_log, bigram_param_log, text, vocabulary))
validation_accuracy_bigram = model_accuracy(df_validate_processed, lambda text: nb_bigram_classifier(cond_param_log, class_param_log, bigram_param_log, text, vocabulary))
print('Calculated Training Accuracy for processed data with bigram feature')

# Training using Naive Bayes with length featuring
interval_size = 5
length_param_log = nb_length_training(df_train_processed, vocabulary, interval_size)

# Model Accuracy length
training_accuracy_length = model_accuracy(df_train_processed, lambda text: nb_length_classifier(cond_param_log, class_param_log, length_param_log, interval_size, text, vocabulary))
validation_accuracy_length = model_accuracy(df_validate_processed, lambda text: nb_length_classifier(cond_param_log, class_param_log, length_param_log, interval_size, text, vocabulary))
print('Calculated Training Accuracy for processed data with length feature')

# Model Accuracy length + bigram
training_accuracy_bigram_length = model_accuracy(df_train_processed, lambda text: nb_bigram_length_classifier(cond_param_log, class_param_log, bigram_param_log, length_param_log, interval_size, text, vocabulary))
validation_accuracy_bigram_length = model_accuracy(df_validate_processed, lambda text: nb_bigram_length_classifier(cond_param_log, class_param_log, bigram_param_log, length_param_log, interval_size, text, vocabulary))
print('Calculated Training Accuracy for processed data with bigram and length feature')


# Part(f)
# Processing Validation data
colName_text_split = 'Tweet'
df_validate_da = read_data('../data/Domain_Adaptation/Twitter_validation.csv')
df_validate_da = df_validate_da.rename(columns={colName_text_split : colName_text})
df_validate_da[colName_text] = df_validate_da[colName_text].apply(simplify_text).apply(stem_text).apply(remove_stop_words)
split_perc = [1, 2, 5, 10, 25, 50, 100]
validation_accuracy_da = []
validation_accuracy_split = []

for split_val in split_perc:
    # Processing Split data
    df_train_split = read_data(f'../data/Domain_Adaptation/Twitter_train_{split_val}.csv')
    df_train_split = df_train_split.rename(columns={colName_text_split : colName_text})
    df_train_split[colName_text] = df_train_split[colName_text].apply(simplify_text).apply(stem_text).apply(remove_stop_words)
    df_train_da = pd.concat([df_train_split, df_train_processed], ignore_index=True, axis = 0)
    
    # Create Vocabulary
    vocabulary_da = create_vocabulary(df_train_da)
    vocabulary_split = create_vocabulary(df_train_split)
    print(f'Vocabulary created for Domain Adaptation split% = {split_val}')

    # Training using Naive Bayes
    class_param_da, cond_param_da = nb_training(df_train_da, vocabulary_da)
    class_param_da_log = np.log10(class_param_da)
    cond_param_da_log = np.log10(cond_param_da)
    class_param_split, cond_param_split = nb_training(df_train_split, vocabulary_split)
    class_param_split_log = np.log10(class_param_split)
    cond_param_split_log = np.log10(cond_param_split)
    print(f'Trained Naive Bayes with processed data for Domain Adaptation split% = {split_val}')

    # Model Accuracy
    validation_accuracy_da.append(model_accuracy(df_validate_da, lambda text: nb_classifier(cond_param_da_log, class_param_da_log, text, vocabulary_da)))
    validation_accuracy_split.append(model_accuracy(df_validate_da, lambda text: nb_classifier(cond_param_split_log, class_param_split_log, text, vocabulary_split)))
    print(f'Calculated Training Accuracy for processed data for Domain Adaptation split% = {split_val}')

#Plotting Accuracies
fig = plt.figure()
ax = fig.add_subplot()
ax.set_title('Domain Adaptation Accuracies')
ax.plot(split_perc, validation_accuracy_da, label = 'Trained by Source and Target', color = 'blue')
ax.plot(split_perc, validation_accuracy_split, label = 'Trained by only Target', color = 'red')
ax.set_xlabel('Split Percent')
ax.set_ylabel('Accuracy')
ax.legend()
fig.savefig('Domain Adaptation Accuracy')


# Create report
create_report('report.txt')