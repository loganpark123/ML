import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import category_encoders as ce
from sklearn.model_selection import train_test_split

data = 'car_evaluation.csv'
df = pd.read_csv(data, header=None)
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
X = df.drop(['class'], axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

def make_tree(depth):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=0)
    model.fit(X_train, y_train)
    y_pred_gini = model.predict(X_test)
    y_pred_train_gini = model.predict(X_train)
    return accuracy_score(y_train, y_pred_train_gini), accuracy_score(y_test, y_pred_gini)

depths = list(range(2,12))
test_accuracy = []
train_accuracy = []

print('depth | train accuracy | Test accuracy')
for x in depths:
    train_accuracy.append(make_tree(x)[0])
    test_accuracy.append(make_tree(x)[1])
    print(x, train_accuracy[x-2], test_accuracy[x-2])
    
plt.plot(depths,test_accuracy, label="Test accuracy")
plt.plot(depths,train_accuracy, label='Training accuracy')
plt.xlabel("Maximum depth of tree")
plt.ylabel("Accuracy")
plt.title("Tuning Maximum Tree Depth")
plt.legend()