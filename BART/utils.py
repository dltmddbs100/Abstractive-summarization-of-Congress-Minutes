import os
import glob
import json
from tqdm import tqdm, trange
from functools import reduce

import numpy as np
import pandas as pd

import re
import nltk
nltk.download('punkt')
from nltk import sent_tokenize


def process_basic(data,context_column):
  for i in range(0,len(data)):
    
    data['title'][i]=re.sub(' +',' ',data['title'][i])
    data['title'][i]=re.sub('본 *회 *의 *회 *의 *록','본회의',data['title'][i])
    data['title'][i]=re.sub('제 *(\d+) *차',r'제\1차',data['title'][i])
    data['title'][i]=re.sub('\(\d+\.\d+.\d+.\)*( *\w{1}요일)*\)*','',data['title'][i])
    data[context_column][i]=re.sub(' +',' ',data[context_column][i]).replace('\u2003','').replace('\\U000f','')

  return data


def process_detail(data,context_column):
  for i in range(0,len(data)):

    sp=sent_tokenize(data[context_column][i])
    a=pd.DataFrame(sp)
    k=a[(a[0].str.contains('감사합니다'))|(a[0].str.contains('질의하실'))|(a[0].str.contains('수고하셨습니다'))|(a[0].str.contains('수고하여 주시기 바랍니다'))|
        (a[0].str.contains('질의ㆍ답변'))|(a[0].str.contains('생략하겠습니다'))|(a[0].str.contains('자세한 내용은'))|(a[0].str.contains('산회를 선포합니다'))|
        (a[0].str.contains('이상입니다'))|(a[0].str.contains('별다른'))|(a[0].str.contains('수고 하셨습니다'))|(a[0].str.contains('수고 많이 하셨습니다'))].index.values
    data.at[i,context_column]=' '.join(np.delete(sp,k))

    data.at[i,context_column]=re.sub('[((위){1}|(의){1})]원 *여러분[,!]* ((다른 의견이)|(이의(가)*)) (없)*(있)*(으십니까)*(습니까)*(으신지요)*\?*','',data[context_column][i])
    data.at[i,context_column]=re.sub('\( *[{「『“]{1}((없)|(있))습니다[」』”}]{1} *하는 [의위]{1}원 있음[. ]*\) *','',data[context_column][i])
    data.at[i,context_column]=re.sub('[((위){1}|(의){1})]원님들[ ,!]* ((다른 의견이)|(이의(가)*)) (없)*(있)*(으십니까)*(습니까)*(으신지요)*\?*','',data[context_column][i])
    data.at[i,context_column]=re.sub('\( *더 *이상 *이의가 *없음 *\)','',data[context_column][i])
    data.at[i,context_column]=re.sub('\( *없습니다 *하는 *의원 *있음 *\)','',data[context_column][i])
    data.at[i,context_column]=re.sub('그러면 *바로 *의결하고자 *합니다.','',data[context_column][i])
    data.at[i,context_column]=re.sub(' ((이상으로)|(이상)),* *(((검토)*보고를)|((제안)*설명*을)) *마치겠습니다.','',data[context_column][i])
    data.at[i,context_column]=re.sub(' +',' ',data[context_column][i])

  return data


# Election case preprocessing
def election(data, column):

  election_ind=data[data[column].str.contains('당선되었음')][column].index

  for i in election_ind:

    sp=sent_tokenize(data[column][i])
    a=pd.DataFrame(sp)
    k=a[(a[0].str.contains('(총 *[\d]+표)|(총 *투표수)|(당선 *되었)|(임기)|(투표 *결과)|\d표|(상정)',regex=True))].index.values
    a=reduce(lambda z, y :z + y, a.iloc[k].values.tolist())
    a=' '.join(a)
    
    data.at[i,column]=a
    
  return data


# Budget Closing Case Preprocessing
def monetary(data, column,tokenizer):

  yebibi_ind=data[(data[column].str.contains('(추가 *경정 *예산안 *제안 *설명)|(예비비 *지출 *승인)|(세입[ㆍ ]*세출 *결산)|(기금 *운용 *계획)',regex=True))][column].index

  for i in yebibi_ind:
    sp=sent_tokenize(data[column][i])
    a=pd.DataFrame(sp)
    if len(tokenizer.encode(data[column][i]))>=300:
      if (a.iloc[:3][0].str.contains('(추가 *경정 *예산안 *제안 *설명)|(예비비 *지출 *승인)|(세입[ㆍ ]*세출 *결산)|(기금 *운용 *계획)').sum()>0) & (a.iloc[:3][0].str.contains('구성 *결의 *안').sum()==0):
        k=a[(a[0].str.contains('기간|의결|가결|의원|위원|상정|(제안 *설명)|이의',regex=True))].index.values
        k=k.tolist()
        k.extend((0,1,2,3,4))
        k=list(sorted(set(k)))
        a=reduce(lambda z, y :z + y, a.iloc[k].values.tolist())
        a=' '.join(a)

        data.at[i,column]=a
        
  return data
