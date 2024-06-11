import numpy as np
import scipy.spatial
from nltk.tokenize import word_tokenize


class SentenceEmbeddings:
    def __init__(self):
        pass

    def sowe(self, array):

        return sum(array)

    def mowe(self, array):

        return sum(array)/len(array)

    def gpt_sowe(self, array):

        return [sum(i) for i in zip(*array)]
    

class SimilarityFunctions:

    def __init__(self, word_array_1, word_array_2):

        self.word_array_1 = word_array_1
        self.word_array_2 = word_array_2

    def get_cosine_similarity(self,u,v):
        return 1-scipy.spatial.distance.cosine(u,v)

    def __cosine_similarity_matrix(self):
        '''
        Creates a matrix depicting the cosine distances between the words of two sentences
        returns: array
          Similarity matrix of words in two sentences
        '''
        matrix = np.zeros((len(self.word_array_1), len(self.word_array_2)))

        for i in range(0, len(self.word_array_1)):
            for j in range(0, len(self.word_array_2)):
                matrix[i][j] = self.get_cosine_similarity(self.word_array_1[i], self.word_array_2[j])
        return matrix.T


    def get_similar_words(self, sentence_1, sentence_2):
        '''Prints similar word from second sentence for each word in the first sentence
        returns: list of similar words
        '''

        token_1 = word_tokenize(sentence_1)
        token_2 = word_tokenize(sentence_2)

        similarity_matrix = self.__cosine_similarity_matrix()

        similar_word_dict = {}
        for row in range(0, len(similarity_matrix[0])):

            min_val = min(similarity_matrix.T[row])  # Here min value of transpose is found. To understand it print similarity matrix and find the logic
            index = (np.where(similarity_matrix.T[row] == min_val))[0]
            similar_word_list = []

            for i in range(0, len(index)):
                similar_word_list.append(token_2[index[i]])
            similar_word_dict[token_1[row]] = similar_word_list

        print('Similar words in two sentences are :', similar_word_dict)