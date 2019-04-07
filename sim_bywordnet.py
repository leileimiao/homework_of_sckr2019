# -*- coding: utf-8 -*-
'''
using wordnet to calculate the similarity of two words
path_similarity
Leacock-Chodorow Similarity
Wu-Palmer Similarity
res_similarity
jcn_similarity
lin_similarity
'''

from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
import pandas as pd
import numpy as np
import warnings       
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from scipy import stats
 
from sklearn.preprocessing import MinMaxScaler,Imputer 

from sklearn.impute import SimpleImputer

words = pd.read_csv("MTURK-771.csv")

wordsList=np.array(words.iloc[:,[0,1]])

simScore=np.array(words.iloc[:,[2]])

page_ScoreList=np.zeros( (len(simScore),1) )

Lch_ScoreList=np.zeros( (len(simScore),1) )

Wp_ScoreList=np.zeros( (len(simScore),1) )

Res_ScoreList=np.zeros( (len(simScore),1) )

Jcn_ScoreList=np.zeros( (len(simScore),1) )

Lin_ScoreList=np.zeros( (len(simScore),1) )

def multi_by_wordnet(wordlist):

    brown_ic = wordnet_ic.ic('ic-brown.dat')

    semcor_ic = wordnet_ic.ic('ic-semcor.dat')

    for i, wordpairs in enumerate(wordlist):
        
        page_count = 0

        syns1 = wordnet.synsets(wordpairs[0])  # the list of synsets

        syns2 = wordnet.synsets(wordpairs[1])

        for word1 in syns1:

            for word2 in syns2 :
                
                try:

                    path_score = word1.path_similarity(word2)

                except Exception :

                    pass

                try:
                        Wp_score = word1.wup_similarity(word2)

                except Exception:
                    
                    pass
                try:
                        Lch_score = word1.lch_similarity(word2)

                except Exception:
                    
                    pass
                try:
                        Res_score= word1.res_similarity(word2,brown_ic)

                except Exception:
                    
                    pass
                try:
                        Jcn_score = word1.jcn_similarity(word2,brown_ic)

                except Exception:
                    
                    pass

                try:
                        Lin_score = word1.lin_similarity(word2,semcor_ic)

                except Exception:
                    
                    pass

             
                if path_score is not None:

                    #print(path_score)

                    page_ScoreList[i,0]+= path_score

                    Lch_ScoreList[i,0]+=Lch_score

                    Wp_ScoreList[i,0]+=Wp_score

                    Res_ScoreList[i,0]+=Res_score

                    Jcn_ScoreList[i,0]+=Jcn_score

                    Lin_ScoreList[i,0]+=Lin_score

                    page_count+=1
        
        if page_count!=0:

            page_ScoreList[i,0] = page_ScoreList[i,0]/(page_count*1.0)

            Lch_ScoreList[i,0] = Lch_ScoreList[i,0]/(page_count*1.0)

            Wp_ScoreList[i,0] = Wp_ScoreList[i,0]/(page_count*1.0)

            Res_ScoreList[i,0] = Res_ScoreList[i,0]/(page_count*1.0)

            Jcn_ScoreList[i,0] = Jcn_ScoreList[i,0]/(page_count*1.0)

            Lin_ScoreList[i,0] = Lin_ScoreList[i,0]/(page_count*1.0)
            #print(page_ScoreList[i,0])
        

    #print(page_ScoreList)

    set_imp=Imputer(missing_values='NaN', strategy='mean', axis=0)
    #SimpleImputer(missing_values='NaN', strategy='mean')

    imp_pageList=set_imp.fit_transform(page_ScoreList)

    imp_LchList = set_imp.fit_transform(Lch_ScoreList)

    imp_WpList =  set_imp.fit_transform(Wp_ScoreList)

    imp_ResList = set_imp.fit_transform(Res_ScoreList)

    imp_JcnList = set_imp.fit_transform(Jcn_ScoreList)

    imp_LinList = set_imp.fit_transform(Lin_ScoreList)


    set_mms=MinMaxScaler(feature_range=(0.0,10.0))

    Mms_page_List=set_mms.fit_transform(imp_pageList)

    Mms_LchList = set_mms.fit_transform(imp_LchList)

    Mms_WpList = set_mms.fit_transform(imp_WpList)

    Mms_ResList = set_mms.fit_transform(imp_ResList)

    Mms_JcnList = set_mms.fit_transform(imp_JcnList)

    Mms_LinList = set_mms.fit_transform(imp_LinList)

    new_simScore = set_mms.fit_transform(simScore)


    (coef_page, pvalue_page)=stats.spearmanr(new_simScore, Mms_page_List)

    (coef_Lch, pvalue_Lch)=stats.spearmanr(new_simScore, Mms_LchList)

    (coef_Wp, pvalue_Wp)=stats.spearmanr(new_simScore, Mms_WpList)

    (coef_Res, pvalue_Res)=stats.spearmanr(new_simScore, Mms_ResList)

    (coef_Jcn, pvalue_Jcn)=stats.spearmanr(new_simScore, Mms_JcnList)

    (coef_Lin, pvalue_Lin)=stats.spearmanr(new_simScore, Mms_LinList)
    
    print(coef_page, pvalue_page)

    print(coef_Lch, pvalue_Lch)

    print(coef_Wp, pvalue_Wp)

    print(coef_Res, pvalue_Res)

    print(coef_Jcn, pvalue_Jcn)

    print(coef_Lin, pvalue_Lin)
 
    submitData=np.hstack( (wordsList, simScore, Mms_page_List,Mms_LchList,Mms_WpList,Mms_ResList,Mms_JcnList,Mms_LinList) )

    (pd.DataFrame(submitData)).to_csv("wordnet.csv", index=False, header=["Word1","Word2","OriginSimilarity","PageSimilarity","Leacock-Chodorow Similarity","Wu-Palmer Similarity","res_similarity","jcn_similarity","lin_similarity"])

if __name__ == '__main__':
    
    multi_by_wordnet(wordsList)

            