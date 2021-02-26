import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import f1_score

sns.set_style('darkgrid')

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


def model_f():

    global fpr, tpr

    model = RandomForestClassifier(random_state=12345)
    model.fit(x_train, y_train)

    result = model.predict(x_valid)

    probs = model.predict_proba(x_valid)
    probs = probs[:, 1]

    print("F1 Score {0}".format(f1_score(y_valid, result)))
    print("AUC Score {0}".format(roc_auc_score(y_valid, probs)))

    fpr, tpr, thresholds = roc_curve(y_valid, probs)

model_f()

plt.figure(figsize=(12, 9))

plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

plt.title('Кривая ROC', fontsize=25)
plt.xlabel('Доля ложноположительных исходов', fontsize=20)
plt.ylabel('Доля истинноположительных исходов', fontsize=20)

plt.legend()
#plt.savefig('Model_3')
plt.show()

