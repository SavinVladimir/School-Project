import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import f1_score

sns.set_style("darkgrid")

data = pd.read_csv('data/Churn_Modelling.csv')

print(data.head())
print()
print('Количество строк и колонок:')
print(data.shape)
print()
print('Типа данных каждого столбца:')
print(data.dtypes)
print()
print('Краткая статистика:')
print(data.describe())
print()
print('Сумма пропущеных значений в каждой столбце:')
print(data.isnull().sum())


def Gender_pie():
    x = data['Gender'].value_counts()

    plt.rcParams['figure.figsize'] = (8, 8)
    plt.pie(x, colors=['#6FEF8D', '#FFA6A3'],
            explode=[0, 0.1],
            labels=['Женщины', 'Мужчины'],
            shadow=True,
            autopct='%.2f%%')

    plt.title('Пол', fontsize=25)
    plt.legend(fontsize=15)
    # plt.savefig('Gender_pie')
    plt.show()


# Gender_pie()


def Gender_hist():
    plt.figure(figsize=(12, 9))

    x = ['Female', 'Male']

    y1 = int((data.Gender == 'Male').mean() * 10000)
    y2 = int((data.Gender == 'Female').mean() * 10000)

    plt.bar(x[0], y1, color='#5158DD', label='Мужчины')
    plt.bar(x[1], y2, color='#FAB832', label='Женщины')

    label = [y1, y2]
    plt.text(0, 5457, s=label[0], fontsize=15)
    plt.text(1, 4543, s=label[1], fontsize=15)

    plt.xlabel('Пол', fontsize=20)
    plt.ylabel('Количество клиентов', fontsize=20)
    plt.title('Пол клиентов', fontsize=25)

    plt.legend(fontsize=15)
    # plt.savefig('Gender_hist')
    plt.show()


# Gender_hist()


def Exited_hist():
    plt.figure(figsize=(12, 9))

    x = ['Refused', 'No refused']

    y1 = int((data.Exited == 0).mean() * 10000)
    y2 = int((data.Exited == 1).mean() * 10000)

    plt.bar(x[0], y1, color='#5158DD', label='Не октазались')
    plt.bar(x[1], y2, color='#FAB832', label='Отказались')

    label = [y1, y2]
    plt.text(0, 7963, s=label[0], fontsize=15)
    plt.text(1, 2037, s=label[1], fontsize=15)

    plt.xlabel('Тип клиента', fontsize=20)
    plt.ylabel('Количество клиентов', fontsize=20)
    plt.title('Клиенты', fontsize=25)

    plt.legend(fontsize=15)
    plt.savefig('Exited_hist')
    plt.show()


# Exited_hist()


def Country_hist():
    plt.figure(figsize=(12, 9))

    x = ['France', 'Spain', 'Germany']
    y1 = int((data.Geography == 'France').mean() * 10000)
    y2 = int((data.Geography == 'Spain').mean() * 10000)
    y3 = int((data.Geography == 'Germany').mean() * 10000)

    plt.bar(x[0], y1, color='#5158DD', label='Франция')
    plt.bar(x[1], y2, color='#FAB832', label='Испания')
    plt.bar(x[2], y3, color='#FC6441', label='Германия')

    label = [y1, y2, y3]
    plt.text(0, 5014, s=label[0], fontsize=15)
    plt.text(1, 2477, s=label[1], fontsize=15)
    plt.text(2, 2509, s=label[2], fontsize=15)

    plt.xlabel('Страна', fontsize=20)
    plt.ylabel('Количество клиентов', fontsize=20)
    plt.title('Кол-во пользователей в странах', fontsize=25)

    plt.legend(fontsize=15)
    # plt.savefig('Country_hist')
    plt.show()


# Country_hist()


def Tenure_hist():
    plt.figure(figsize=(12, 9))

    y1 = data[data.Exited == 1].Tenure
    y2 = data[data.Exited == 0].Tenure

    plt.hist([y1, y2], rwidth=0.95, color=['#5158DD', '#FC6441'], label=['Покинул банк', 'По прежнему клиент'])

    plt.xlabel('Недвижимость', fontsize=20)
    plt.ylabel('Количество клиентов', fontsize=20)
    plt.title('Визуализация прогнозирования оттока клиентов', fontsize=25)

    plt.legend(fontsize=15)
    # plt.savefig('Tenure_hist')
    plt.show()


# Tenure_hist()


def Credit_Score_hist():
    plt.figure(figsize=(12, 9))

    y1 = data[data.Exited == 1].CreditScore
    y2 = data[data.Exited == 0].CreditScore

    plt.hist([y1, y2], rwidth=0.95, color=['#5158DD', '#FC6441'], label=['Покинул банк', 'По прежнему клиент'])

    plt.xlabel('Ежемесячные платежи', fontsize=20)
    plt.ylabel('Количество клиентов', fontsize=20)
    plt.title('Визуализация прогнозирования оттока клиентов', fontsize=25)

    plt.legend(fontsize=15)
    # plt.savefig('Credit_Score_hist')
    plt.show()


# Credit_Score_hist()


def Age_hist():
    plt.figure(figsize=(12, 9))

    y1 = data[data.Exited == 1].Age
    y2 = data[data.Exited == 0].Age

    plt.hist([y1, y2], rwidth=0.95, color=['#5158DD', '#FC6441'], label=['Покинул банк', 'По прежнему клиент'])

    plt.xlabel('Возраст клиентов', fontsize=20)
    plt.ylabel('Количество клиентов', fontsize=20)
    plt.title('Визуализация прогнозирования оттока клиентов', fontsize=25)

    plt.legend(fontsize=15)
    # plt.savefig('Age_hist')
    plt.show()


# Age_hist()


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

false_positive_rate = None
true_positive_rate = None


def model_score():
    global false_positive_rate, true_positive_rate

    model = RandomForestClassifier(random_state=12345)
    model.fit(x_train, y_train)

    result = model.predict(x_valid)

    probs = model.predict_proba(x_valid)
    probs = probs[:, 1]

    print()
    print('Статистические метрики:')
    print("F1 Score {0}".format(f1_score(y_valid, result)))
    print("AUC Score {0}".format(roc_auc_score(y_valid, probs)))

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_valid, probs)


model_score()

plt.figure(figsize=(12, 9))

plt.plot(false_positive_rate, true_positive_rate, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

plt.title('Оценка классификатора случайного леса', fontsize=25)
plt.xlabel('Доля ложноположительных исходов', fontsize=20)
plt.ylabel('Доля истинноположительных исходов', fontsize=20)

plt.legend()
plt.show()
