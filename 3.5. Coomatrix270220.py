import numpy as np
import pandas as pd

def Vector2CoOMatrix(vector):
    for i in range(vector.shape[0]):
        cur=np.zeros(shape=(vector.shape[0],),dtype=np.dtype('u1'))
        if vector[i]==1:
            for j in range(vector.shape[0]):
                if vector[j]==1:
                    cur[j]=1
        if i==0:
            coo_matrix=cur
        else:
            coo_matrix=np.vstack((coo_matrix,cur))
    return coo_matrix
def CoOlize(vectors):
    for i in range(vectors.shape[0]):
        if i==0:
            output=np.expand_dims(Vector2CoOMatrix(vectors[i,:]),axis=0)
        else:
            output=np.vstack((output,np.expand_dims(Vector2CoOMatrix(vectors[i,:]),axis=0)))
    return output