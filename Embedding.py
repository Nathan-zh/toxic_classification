import numpy as np


class Embedding():
    def __init__(self, typeToLoad = "glove"):
    # This function is simply load the pre-trained embedding dataset to a python dictionary
         if (typeToLoad == "glove"):
              EMBEDDING_FILE = './glove.twitter.27B.25d.txt'

         if (typeToLoad == "glove" or typeToLoad == "fasttext"):
             self.embeddings_index = dict()
             # Transfer the embedding weights into a dictionary by iterating through every line of the file.
             f = open(EMBEDDING_FILE)
             for line in f:
                 # split up line into an indexed array
                 values = line.split()
                 # first index is word
                 word = values[0]
                 # store the rest of the values in the array as a new array
                 coefs = np.asarray(values[1:], dtype='float32')
                 self.embeddings_index[word] = coefs  # 25 dimensions
             f.close()
             print('Loaded %s word vectors.' % len(self.embeddings_index))

    # The input is a string with " " to split up.
    def wordsToVec(self, word):
        output = []
        seq_maxlen = 200
        for m in word:
            word_list = m.split(" ")  # change word string to list
            lenW = len(word_list)
            word_T = np.zeros([25, seq_maxlen])  # An empty nparray for vectors
            if lenW > seq_maxlen:
                for i in range(seq_maxlen):
                    if word_list[i] in self.embeddings_index:
                        word_T[:, i] = np.array(self.embeddings_index[word_list[i]])
                    else:
                        continue

            else:
                j = seq_maxlen - 1
                for i in range(lenW-1, -1, -1):
                    if word_list[i] in self.embeddings_index:
                        word_T[:, j - (lenW -1 -i)] = self.embeddings_index[word_list[i]]
                        j -= 1
                    else:
                        continue
            output.append(word_T.T)

        return output




