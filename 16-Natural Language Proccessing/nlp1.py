# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 11:24:22 2021

@author: kdrer
"""

import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv', error_bad_lines=False)

import re
import nltk 

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

#Preprocessing (Önişleme)
derlem = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)