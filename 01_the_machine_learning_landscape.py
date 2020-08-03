#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
keras.__version__


# In[2]:


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# 예제 1-1 사이킷런을 이용한 선형 모델의 훈련과 실행

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


# In[4]:


cd datasets/lifesat


# In[5]:


# 데이터 적재
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=",")
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=",", delimiter="\t",
                             encoding="latin1", na_values="n/a")


# In[6]:


# 데이터 준비
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]


# In[7]:


# 데이터 시각화
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()


# In[8]:


# 선형 모델 선택
model = sklearn.linear_model.LinearRegression()


# In[9]:


# 모델 훈련
model.fit(X, y)


# In[10]:


# 키프로스에 대한 예측 만들기 
X_new = [[22587]] # 키프로스 1인당 GDP
print(model.predict(X_new)) # 결과 [[5.96242338]]


# k-최근접 이웃 회귀로 표현

# In[11]:


# 선형 모델 선택
import sklearn.neighbors
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)


# In[12]:


# 모델 훈련
model.fit(X, y)


# In[13]:


# 키프로스에 대한 예측 만들기 
X_new = [[22587]] # 키프로스 1인당 GDP
print(model.predict(X_new)) # 결과 [[5.96242338]]


# In[ ]:




