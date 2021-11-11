#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd

# modeling imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


# In[15]:


#Load the data
df = pd.read_csv('bank-direct-marketing-campaigns.csv')
df.head()


# In[16]:


#convert the catogeral to num

cleanup_nums = {"y":     {"no": 0, "yes": 1},
                "job": {"housemaid": 0,
                       "services": 1,
                       "admin": 2,
                       "technician": 3,
                       "housemaid": 4,
                       "retired": 5,
                       "unemployed": 6,
                       "self-employed": 7,
                       "unknown": 8,
                       "management": 9,
                       "entrepreneur": 10,
                       "student": 11},
                "marital":{"married": 0,
                          "single": 1,
                          "divorced":2,
                          "unknown": 3},
                "education":{"basic.4y": 0,
                             "high.school": 1,
                             "basic.6y": 2,
                             "basic.9y": 3,
                             "university.degree": 4,
                             "professional.course": 5,
                             "unknown": 6,
                            "illiterate":7},
                "default":{"unknown": 0,
                           "no": 1,
                           "yes":2
                    
                },
                "housing":{"unknown":0, 
                          "no":1,
                          "yes":2},
                
                "loan":{"unknown":0, 
                          "no":1,
                          "yes":2},
                
                "contact":{"telephone":0,
                          "cellular": 1},
                 "month":{"may":0,
                         "jun":1,
                         "jul":2,
                         "aug":3,
                         "sep":4,
                         "oct":5,
                          "nov":6,
                          "apr":7,
                          "mar":8,
                          "dec":9
                          
                    },
                "day_of_week":{"mon": 0,
                             "tue": 1,
                             "wed": 2,
                             "thu": 3,
                             "fri": 4  
                    
                },
                  "poutcome":{"nonexistent":0,
                             "failure": 1,
                             "success":2},
               
               
               
               
               
               
               
               }


# In[17]:


obj_df = df.replace(cleanup_nums)
obj_df.head()


# In[18]:


obj_df.dtypes


# In[19]:


y= obj_df['y']


# In[20]:


y.head()


# In[21]:


x= obj_df[['age','job','marital','education','default',
         'housing','loan','contact','month','day_of_week',
         'campaign','pdays','previous','poutcome','emp.var.rate',
        'cons.price.idx', 'cons.conf.idx', 'euribor3m',
        'nr.employed']]


# In[22]:


# split the data 80, 20
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=42)


# In[23]:


x.head()


# In[24]:


X_train.describe()


# In[25]:


y_train.describe()


# In[26]:



#fit the model and calculate the log
lr_model_all = LogisticRegression(C=10000)
lr_model_all.fit(X_train, y_train)
y_prob_pred_test = lr_model_all.predict_proba(X_test)
print(log_loss(y_test,y_prob_pred_test))


# In[ ]:




