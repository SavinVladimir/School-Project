import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import f1_score

sns.set_style("darkgrid")

data = pd.read_csv('../data/Churn_Modelling.csv')


def data_processing():
    global data
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    data = pd.get_dummies(data)
    data = data.rename(columns={'Gender_Female': 'Female',
                                'Gender_Male': 'Male',
                                'Geography_France': 'France',
                                'Geography_Germany': 'Germany',
                                'Geography_Spain': 'Spain'})


data_processing()


def data_split(df, train_percent=0.6, validate_percent=0.2, seed=None):

    np.random.seed(seed)

    p = np.random.permutation(df.index)
    m = len(df.index)

    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end

    train = df.iloc[p[:train_end]]
    validate = df.iloc[p[train_end:validate_end]]
    test = df.iloc[p[validate_end:]]

    return train, validate, test


data_train, data_valid, data_test = data_split(data, seed=12345)

x_train = data_train.drop(['Exited'], axis=1)
y_train = data_train['Exited']

x_valid = data_valid.drop(['Exited'], axis=1)
y_valid = data_valid['Exited']

x_test = data_test.drop(['Exited'], axis=1)
y_test = data_test['Exited']

def model_1():

    global fpr_1, tpr_1

    model = LogisticRegression(random_state=12345)
    model.fit(x_train, y_train)

    result = model.predict(x_valid)

    probs = model.predict_proba(x_valid)
    probs = probs[:, 1]

    print('Модель - 1')
    print("F1-мера {0}".format(f1_score(y_valid, result)))
    print("Показатель AUC {0}".format(roc_auc_score(y_valid, probs)))

    fpr_1, tpr_1, thresholds = roc_curve(y_valid, probs)

model_1()


def model_2():

    global fpr_2, tpr_2

    model = DecisionTreeClassifier(random_state=12345)
    model.fit(x_train, y_train)

    result = model.predict(x_valid)

    probs = model.predict_proba(x_valid)
    probs = probs[:, 1]

    print('Модель - 2')
    print("F1-мера {0}".format(f1_score(y_valid, result)))
    print("Показатель AUC {0}".format(roc_auc_score(y_valid, probs)))

    fpr_2, tpr_2, thresholds = roc_curve(y_valid, probs)

model_2()


def model_3():

    global fpr_3, tpr_3

    model = RandomForestClassifier(random_state=12345)
    model.fit(x_train, y_train)

    result = model.predict(x_valid)

    probs = model.predict_proba(x_valid)
    probs = probs[:, 1]

    print('Модель - 3')
    print("F1-мера {0}".format(f1_score(y_valid, result)))
    print("Показатель AUC {0}".format(roc_auc_score(y_valid, probs)))

    fpr_3, tpr_3, thresholds = roc_curve(y_valid, probs)

model_3()


plt.figure(figsize=(12, 9))

plt.plot(fpr_1, tpr_1, color='#423FA5', label='Модель - 1')
plt.plot(fpr_2, tpr_2, color='#FFAF0F', label='Модель - 2')
plt.plot(fpr_3, tpr_3, color='#FD4F57', label='Модель - 3')

plt.plot([0, 1], [0, 1], color='#33261E', linestyle='--')

plt.title('Кривая ROC', fontsize=25)
plt.xlabel('Доля ложноположительных исходов', fontsize=20)
plt.ylabel('Доля истинноположительных исходов', fontsize=20)

plt.legend()
#plt.savefig('AllModels')
plt.show()
