import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from PIL import Image



st.header('''Credit Risk Analysis and Prediction using LightGBM''')
st.write("""
This app is a tool that can predict a person's loan repayment capability using LightGBM, one of the most advanced gradient boosting algorithms
around.
Data obtained from the UCI Machine Learning Repository.
""")

default = pd.read_csv('UCI_Credit_Card.csv', index_col="ID")
default.rename(columns=lambda x: x.lower(), inplace=True)
# Base values: female, other_education, not_married
default['grad_school'] = (default['education'] == 1).astype('int')
default['university'] = (default['education'] == 2).astype('int')
default['high_school'] = (default['education'] == 3).astype('int')
default.drop('education', axis=1, inplace=True)

default['male'] = (default['sex']==1).astype('int')
default.drop('sex', axis=1, inplace=True)

default['married'] = (default['marriage'] == 1).astype('int')
default.drop('marriage', axis=1, inplace=True)
pay_features = ['pay_0','pay_2','pay_3','pay_4','pay_5','pay_6']
for p in pay_features:
    default.loc[default[p]<=0, p] = 0
target_name = 'default.payment.next.month'
X = default.drop('default.payment.next.month', axis=1)
robust_scaler = RobustScaler()
X = robust_scaler.fit_transform(X)
y = default[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123, stratify=y)

# For pay features if the <= 0 then it means it was not delayed
pay_features = ['pay_0','pay_2','pay_3','pay_4','pay_5','pay_6']
for p in pay_features:
    default.loc[default[p]<=0, p] = 0

default.rename(columns={'default payment next month':'default'}, inplace=True)

import lightgbm as lgb
d_train = lgb.Dataset(X_train, label=y_train)

lgbm_params = {'learning_rate': 0.039, 'boosting_type': 'gbdt',
                'objective': 'binary',
               'metric': ['auc', 'binary_logloss'],
               'num_leaves': 150,
               'max_depth': 18}

clf = lgb.train(lgbm_params, d_train, 100)

y_pred_lgbm = clf.predict(X_test)

st.sidebar.header('User Input Features')


limit_bal=st.sidebar.text_input('Credit Amount',500)
age=st.sidebar.text_input('Age','25')
bill_amt1=st.sidebar.text_input('Previous Bill Amount 1',100)
bill_amt2=st.sidebar.text_input('Previous Bill Amount 2',0)
bill_amt3=st.sidebar.text_input('Previous Bill Amount 3',0)
bill_amt4=st.sidebar.text_input('Previous Bill Amount 4',0)
bill_amt5=st.sidebar.text_input('Previous Bill Amount 5',0)
bill_amt6=st.sidebar.text_input('Previous Bill Amount 6',100)
pay_amt1=st.sidebar.text_input('Paid Amount 1',100)
pay_amt2=st.sidebar.text_input('Paid Amount 2',0)
pay_amt3=st.sidebar.text_input('Paid Amount 3',0)
pay_amt4=st.sidebar.text_input('Paid Amount 4',0)
pay_amt5=st.sidebar.text_input('Paid Amount 5',0)
pay_amt6=st.sidebar.text_input('Paid Amount 6',100)
sex=st.sidebar.selectbox('Sex',(1,0))
grad_school=st.sidebar.selectbox('Grad School',(0,1))
university=st.sidebar.selectbox('University',(0,1))
high_school=st.sidebar.selectbox('High School',(0,1))
married=st.sidebar.selectbox('Marital Status',(1,2,3))
pay_0=st.sidebar.selectbox('Payment Status 1',(-1,1,2,3,4,5,6,7,8,9))
pay_2=st.sidebar.selectbox('Payment Status 2',(-1,1,2,3,4,5,6,7,8,9))
pay_3=st.sidebar.selectbox('Payment Status 3',(-1,1,2,3,4,5,6,7,8,9))
pay_4=st.sidebar.selectbox('Payment Status 4',(-1,1,2,3,4,5,6,7,8,9))
pay_5=st.sidebar.selectbox('Payment Status 5',(-1,1,2,3,4,5,6,7,8,9))
pay_6=st.sidebar.selectbox('Payment Status 6',(-1,1,2,3,4,5,6,7,8,9))
data = {
    'limit_bal':limit_bal,
    'age':age,
    'bill_amt1':bill_amt1,
    'bill_amt2':bill_amt2,
    'bill_amt3':bill_amt3,
    'bill_amt4':bill_amt4,
    'bill_amt5':bill_amt5,
    'bill_amt6':bill_amt6,
    'pay_amt1':pay_amt1,
    'pay_amt2':pay_amt2,
    'pay_amt3':pay_amt3,
    'pay_amt4':pay_amt4,
    'pay_amt5':pay_amt5,
    'pay_amt6':pay_amt6,
    'sex':sex,
    'grad_school':grad_school,
    'university':university,
    'high_school':high_school,
    'married':married,
    'pay_0':pay_0,
    'pay_2':pay_2,
    'pay_3':pay_3,
    'pay_4':pay_4,
    'pay_5':pay_5,
    'pay_6':pay_6
}
from collections import OrderedDict
new_customer = OrderedDict([('limit_bal', limit_bal), ('age', age), ('bill_amt1', bill_amt1),
                                ('bill_amt2', bill_amt2), ('bill_amt3',
                                                           bill_amt3), ('bill_amt4', bill_amt4),
                                ('bill_amt5', bill_amt5), ('bill_amt6',
                                                           bill_amt6), ('pay_amt1', pay_amt1), ('pay_amt2', pay_amt2),
                                ('pay_amt3', 0), ('pay_amt4',
                                                  0), ('pay_amt5', 0), ('pay_amt6', 0),
                                ('sex', sex), ('grad_school', grad_school), ('university',
                                                                             university), ('high_school', high_school),
                                ('married', married), ('pay_0',
                                                       pay_0), ('pay_2', pay_2), ('pay_3', pay_3),
                                ('pay_4', pay_4), ('pay_5', pay_5), ('pay_6', pay_6)])

new_customer = pd.Series(new_customer)
data = new_customer.values.reshape(1, -1)
data = robust_scaler.transform(data)
st.header('Prediction from LightGBM Model')

prob = clf.predict(data)[0]
if (prob >= 0.5):
    st.write('Will default')
else:
    st.write('Will pay')

st.header('''Comparative Study Results''')
st.write('This is how the algorithms compare:')
study = pd.read_csv('comparisondata.csv')
algorithms = ['LightGBM', 'XGBoost', 'Logistic Regression', 'CatBoost']
fig = go.Figure(data=[
    go.Bar(name='Accuracy', x=algorithms, y=[
        85.5556, 82.4222, 81.9778, 82.066667]),
    go.Bar(name='Precision', x=algorithms, y=[
        78.798, 70.4, 69.8276, 65.766739]),
    go.Bar(name='Recall', x=algorithms, y=[
        47.4372, 35.3769, 32.5628, 37.202199])
])

fig.update_layout(barmode='group')
st.plotly_chart(fig)
st.table(study)
st.write('We conclude that LightGBM outperforms the other three algorithms and is hence, better suited for this use case. The execution time is faster than that of both XGBoost and Catboost. It is slightly slower than Logistic Regression however it makes up for it through higher accuracy, precision and recall. Precision and recall are the two most important factors in determining how good an algorithm is for predicting defaults. This is because in this particular project, false positives mean people who paid being classed as defaults and false negatives mean the opposite. Having a higher number of false negatives is bad, this can be avoided if the algorithm has better recall.')
st.write('Here are some diagrams: ')

image = Image.open('images/Heatmap.png')
st.image(image, caption='heatmap')
st.write('''The heatmap you see above depicts the correlation between all the categorical features available in the dataset. There are 24 features in total.''')
st.write(
    '● Limit_bal is the feature that depicts the amount given as credit to the client.')
st.write('''● Sex is the next feature. The values under this category are 1 and 2, 1 being male and 2 being
    female.''')
st.write('''● Education has 4 sub-categories, namely, highschool, college/university, graduate school and
    others. This category depicts the educational qualification of an individual client.''')
st.write('''● Marriage depicts the marital status of the clients. There are three subcategories here, namely,
    married, divorced and others.''')
st.write('''● Pay_0 - Pay_5 depicts the loan repayment status of the client starting from April to August
    (in reverse chronological order)''')
st.write('''● Bill_amt1 - Bill_amt6 depicts the bill statement amount in reverse chronological order
    across the 6 month period.''')
st.write('''● Pay_amt1 - Pay_amt6 depicts the previous months repayment status. If the value is -1, it
    means the client had paid duly, if it is 1 it means the client delayed payment by one month
    and so on.''')
st.write('''● Default is the last category. It shows whether that particular client defaulted in repaying their
    loan. 1 means yes, 0 means no.''')
st.write('''There are quite a few insights that can be gained from the heatmap we see above. There is an interesting negative correlation between limit_bal and default. This means that the higher the credit limit is, the lower the default rate is. This is interesting because people tend to fail on loan repayments when the amount is large. The “default” feature correlates the most with the first payment feature pay_0, it means that people who delay their very first payment tend to default. This is logically understandable considering these clients will then have to be repaying a larger amount every subsequent month along with interest, making it incredibly difficult to pay off completely. We can observe that the entire region from pay_0 or pay_1 to pay_6 is darker for defaults. This suggests that payment behaviour is a strong indication of a client’s ability to repay their loan. As understood from the previous point, regular payments can help a client be on the safer side.''')

image1 = Image.open('images/roclgbm.png')
st.image(image1, caption='AUC Curve - LightGBM')
st.write('''From out testing and experimentation, LightGBM came out on top. An AUC score of 0.90 is excellent. This is considerably higher than both XGBoost and Logistic Regression.''')

image2 = Image.open('images/rocxgb.png')
st.image(image2, caption='AUC Curve - XGBoost')
image3 = Image.open('images/roclr.png')
st.image(image3, caption='AUC Curve - Logistic Regression')

st.write('''The diagrams above are called AUC-ROC curves. Before looking at what the graph says, it is important to understand what AUC-ROC means. When it comes to classification problems, especially problems of the multi-class kind, there’s no better measurement for performance. AUC-ROC scores encapsulate the classification ability and quality of an algorithm.
In simple terms, AUC measures the ability of an algorithm to separate or classify. The ability of an algorithm to correctly classify a 0 as a 0 and a 1 as a 1 is measured using AUC. AUC stands for Area Under Curve. ROC stands for Receiver Operating Characteristic.
If the AUC value is high or if the Area Under the Curve in the graph is large, it means that the algorithm identifies or classifies more targets correctly. Hence it is necessary for an algorithm, especially in a use case such as ours, to have the highest AUC value.
LightGBM has an AUC score/value of 0.90 or 90% while the next best algorithm, XGBoost, has an AUC value of 0.78 or 78%. This difference is significant. This essentially means that LightGBM gets its predictions correct 90% of the time. Logistic regression gave us an AUC score of 0.77 or 77%.''')

image4 = Image.open('images/prc.png')
st.image(image4, caption='Precision Recall Curve - LGBM vs XGBoost')

st.write('''Based on these readings, we can say that LightGBM is a very powerful machine learning framework. It is comfortably ahead of XGBoost in terms of precision and recall, two of the most important factors that need to be considered in this use-case.
In terms of execution time, XGBoost is the fastest while CatBoost is the slowest. These are the readings we observed (in ascending order):''')
st.write('''● XGBoost - 2780ms or 2.78s''')
st.write('''● LightGBM - 2926ms or 2.926s''')
st.write('''● Logistic Regression - 6267ms or 6.267s''')
st.write('''● Catboost - 25676ms or 25.676s''')
st.write('''Little to no parameter tuning was performed during the experimentation. We chose to do it this way to find the default performance of each algorithm. We can conclude that LightGBM is the best algorithm overall.''')
