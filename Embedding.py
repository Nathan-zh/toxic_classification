import numpy as np

class Embedding():
    def __init__(self,typeToLoad = "glove"):
    # This function is simply load the pre-trained embedding dataset to a python dictionary
         if (typeToLoad == "glove"):
              EMBEDDING_FILE = './glove.6B.300d.txt'

         if (typeToLoad == "glove" or typeToLoad == "fasttext"):
             self.embeddings_index = dict()
             # Transfer the embedding weights into a dictionary by iterating through every line of the file.
             f = open(EMBEDDING_FILE,encoding="utf-8")
             for line in f:
                 # split up line into an indexed array
                 values = line.split()
                 # first index is word
                 word = values[0]
                      # store the rest of the values in the array as a new array
                 coefs = np.asarray(values[1:], dtype='float32')
                 self.embeddings_index[word] = coefs  # 300 dimensions
             f.close()
             print('Loaded %s word vectors.' % len(self.embeddings_index))

    # The input is a string with " " to split up.
    def wordsToVec(self, word):
        word_list = word.split(" ") # change word string to list
        lenW = len(word_list)
        word_T = [[0] * 300] * 5000 # An empty list for vectors
        for i in range(lenW-1,-1 ,-1):
            if word_list[i] in self.embeddings_index:
                word_T[4999 - i] = self.embeddings_index[word_list[i]]
            else:
                continue


        return word_T




