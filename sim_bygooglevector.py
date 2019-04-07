# -*- coding: utf-8 -*-
'''
using GoogleNews vectors to calculate the similarity of two words
'''
import pandas as pd
import numpy as np
import gensim
from scipy import stats
 
def WordSimibyGoogleNews():

    set = pd.read_csv('MTURK-771.csv')

    data = np.array(set.iloc[ :, [ 0, 1 ] ])

    simScore = np.array(set.iloc[ :, [ 2 ] ])

    Googlescore = np.zeros((len(data), 1))

    Google_model=gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)
    
    Wiki_model = gensim.models.KeyedVectors.load_word2vec_format()
    for i,(word1, word2) in enumerate(data):
        
       
        
        Googlescore[ i, 0 ] = Google_model.similarity(word1,word2)
    
    (coef1, pvalue) = stats.spearmanr(simScore, Googlescore)
    
    submitData = np.hstack((data, simScore, Googlescore))
    
    (pd.DataFrame(submitData)).to_csv("wordsimbypath_GoogleNews.csv", index=False, header=[ "Word1", "Word2", "OriginSimi", "GoogleSimi" ])
    
    print("WordSimibyCS:", 'correlation=', coef1, 'pvalue=', pvalue)
 
if __name__=='__main__':
    
    WordSimibyGoogleNews()
