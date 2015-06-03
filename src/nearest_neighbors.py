from sklearn.neighbors import NearestNeighbors
import numpy as np


class DocNN():

    def __init__(self, D, k=1):
        self.D = D
        self.nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(D)

    def doc_nearest(self, doc_ind):
        distances, indices = self.nbrs.kneighbors(self.D)
        neighbors = indices[doc_ind][1:]
        return self.D[neighbors], neighbors
