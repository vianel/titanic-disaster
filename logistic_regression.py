import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def show_confunsion_matrix(y_test, y_pred):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    class_names = [0, 1]
    fix, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='Blues_r',
                fmt='g')

    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', y=1.1)
    plt.ylabel('Current Label')
    plt.xlabel('Predicted label')

    plt.show()


def main():
    dataset = pd.read_csv('train.csv')
    print(dataset.head(5))

    # Based on some research we gonna drop the insignificantly colums
    data = dataset.drop(['PassengerId', 'Survived', 'Name', 'Sex',
                         'Ticket', 'Cabin', 'Embarked'], axis=1)

    data['Age'] = data['Age'].fillna(data['Age'].median())
    label_encoder = preprocessing.LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(dataset['Sex'])

    x_train, x_test, y_train, y_test = train_test_split(data, dataset.Survived,
                                                        test_size=0.20,
                                                        random_state=0)

    model = LogisticRegression()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("our model classified the data {:.2%} correctly"
          .format(metrics.accuracy_score(y_test, y_pred)))

    show_confunsion_matrix(y_test, y_pred)

    testfile = pd.read_csv('test.csv')

    test_data = testfile.drop(['PassengerId', 'Name', 'Sex',
                               'Ticket', 'Cabin', 'Embarked'], axis=1)

    test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
    testfile['Sex'] = testfile['Sex'].fillna('male')
    test_data['Sex'] = label_encoder.fit_transform(testfile['Sex'])
    test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

    results = model.predict(test_data)

    testfile['Survived'] = results

    survivors = testfile[['PassengerId', 'Survived']]
    survivors.to_csv('survivors.csv', index=False)


if __name__ == '__main__':
    main()
