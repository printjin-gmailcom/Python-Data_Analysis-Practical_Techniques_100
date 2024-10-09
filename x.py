import warnings
warnings.filterwarnings('ignore')



import pandas as pd
survey = pd.read_csv("survey.csv")
print(len(survey))
survey.head()

survey.isna().sum()

survey = survey.dropna()
survey.isna().sum()



survey["comment"] = survey["comment"].str.replace("AA", "")
survey.head()

survey["comment"] = survey["comment"].str.replace("\(.+?\)", "")
survey.head()



survey["length"] = survey["comment"].str.len()
survey.head()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline # 쥬피터노트북의 기능으로 그래프가 노트북 내에 삽입되어 바로 확인 가능
plt.hist(survey["length"])



from konlpy.tag import Twitter #한국어 형태소 분석
twt = Twitter()
text = "형태소분석으로 문장을 분해해보자"
tagging = twt.pos(text)
tagging

words = twt.pos(text)
words_arr = []
for i in words:
    if i == 'EOS': continue
    word_tmp = i[0]
    words_arr.append(word_tmp)
words_arr



text = "형태소분석으로 문장을 분해해보자"
words_arr = []
parts = ["Noun", "Verb"]
words = twt.pos(text)
words_arr = []
for i in words:
    if i == 'EOS' or i == '': continue
    word_tmp = i[0]
    part = i[1]
    if not (part in parts):continue
    words_arr.append(word_tmp)
words_arr



all_words = []
parts = ["Noun"]
for n in range(len(survey)):
    text = survey["comment"].iloc[n]
    words = twt.pos(text)
    words_arr = []
    for i in words:
        if i == "EOS" or i == "": continue
        word_tmp = i[0]
        part = i[1]
        if not (part in parts):continue
        words_arr.append(word_tmp)
    all_words.extend(words_arr)
print(all_words)

all_words_df = pd.DataFrame({"words":all_words, "count":len(all_words)*[1]})
all_words_df = all_words_df.groupby("words").sum()
all_words_df.sort_values("count",ascending=False).head()



stop_words = ["더","수","좀"]
all_words = []
parts = ["Noun"]
for n in range(len(survey)):
    text = survey["comment"].iloc[n]
    words = twt.pos(text)
    words_arr = []
    for i in words:
        if i == "EOS" or i == "": continue
        word_tmp = i[0]
        part = i[1]
        if not (part in parts):continue
        if word_tmp in stop_words:continue
        words_arr.append(word_tmp)
    all_words.extend(words_arr)
print(all_words)

all_words_df = pd.DataFrame({"words":all_words, "count":len(all_words)*[1]})
all_words_df = all_words_df.groupby("words").sum()
all_words_df.sort_values("count",ascending=False).head()



stop_words = ["더","수","좀"]
parts = ["Noun"]
all_words = []
satisfaction = []
for n in range(len(survey)):
    text = survey["comment"].iloc[n]
    words = twt.pos(text)
    words_arr = []
    for i in words:
        if i == "EOS" or i == "": continue
        word_tmp = i[0]
        part = i[1]
        if not (part in parts):continue
        if word_tmp in stop_words:continue
        words_arr.append(word_tmp)
        satisfaction.append(survey["satisfaction"].iloc[n])
    all_words.extend(words_arr)
all_words_df = pd.DataFrame({"words":all_words, "satisfaction":satisfaction, "count":len(all_words)*[1]})
all_words_df.head()

words_satisfaction = all_words_df.groupby("words").mean()["satisfaction"]
words_count = all_words_df.groupby("words").sum()["count"]
words_df = pd.concat([words_satisfaction, words_count], axis=1)
words_df.head()

words_df = words_df.loc[words_df["count"]>=3]
words_df.sort_values("satisfaction", ascending=False).head()

words_df.sort_values("satisfaction").head()



parts = ["Noun"]
all_words_df = pd.DataFrame()
satisfaction = []
for n in range(len(survey)):
    text = survey["comment"].iloc[n]
    words = twt.pos(text)
    words_df = pd.DataFrame()
    for i in words:
        if i == "EOS" or i == "": continue
        word_tmp = i[0]
        part = i[1]
        if not (part in parts):continue
        words_df[word_tmp] = [1]
    all_words_df = pd.concat([all_words_df, words_df] ,ignore_index=True)
all_words_df.head()

all_words_df = all_words_df.fillna(0)
all_words_df.head()



print(survey["comment"].iloc[2])
target_text = all_words_df.iloc[2]
print(target_text)

import numpy as np
cos_sim = []
for i in range(len(all_words_df)):
    cos_text = all_words_df.iloc[i]
    cos = np.dot(target_text, cos_text) / (np.linalg.norm(target_text) * np.linalg.norm(cos_text))
    cos_sim.append(cos)
all_words_df["cos_sim"] = cos_sim
all_words_df.sort_values("cos_sim",ascending=False).head()

print(survey["comment"].iloc[2])
print(survey["comment"].iloc[15])
print(survey["comment"].iloc[24])

