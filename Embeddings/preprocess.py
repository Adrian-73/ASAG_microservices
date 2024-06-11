from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd


def tokenization( question, answer):
    question_tokens = word_tokenize(question)
    answer_tokens = word_tokenize(answer)
    return question_tokens, answer_tokens

def question_demoting( question, answer):

    question_tokens, answer_tokens = tokenization(question, answer)
    demoted_tokens = [word for word in answer_tokens if word not in question_tokens]
    return demoted_tokens

def remove_stop_words( demoted_tokens):

    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in demoted_tokens if not w in stop_words]
    return filtered_sentence

def pair(df):
    questions = df['Question'].values
    keys = df['Key'].values
    answers = ([eval(i) for i in df['Response'].values])
    averages = ([ eval(i) for i in df['Average'].values])
    medians = ([ eval(i) for i in df['Median'].values])
    other_scores = ([ eval(i) for i in df['Other'].values])
    pairs = []
    for i in range(len(questions)):
        
        for j in range(len(answers[i])):
            pairs.append([questions[i], answers[i][j],keys[i],float(averages[i][j]),float(medians[i][j]),float(other_scores[i][j])])
    return pd.DataFrame(pairs,columns=['Question','Response','Key','Averages','Medians','Other_scores'])


def pre_processing(ques, ans):
    """
        Preprocess question and answer. Returns the filtered list of tokens
    :param ques: string
    :param ans: string
    :return: list
        Returns the filtered list after all preprocessing steps
    """
    
    question_demoted = question_demoting(ques, ans)
    filtered_sentence = remove_stop_words(question_demoted)
    return filtered_sentence