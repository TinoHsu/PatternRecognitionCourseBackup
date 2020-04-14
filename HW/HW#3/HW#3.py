import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


# Large scale machine learning and stochastic gradient descent
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent cycles.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value averaged over all
      training samples in each epoch.

        
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
                # print(X, y)
            cost = []
            for xi, target in zip(X, y):
                # print('xi=', xi)
                # print('target=', target)
                cost.append(self._update_weights(xi, target))
                # print(self._update_weights(xi, target))
                # print(self.w_[1:])
                # print(self.w_[0])
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        # print(r)
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

# A function for plotting decision regions
def plot_decision_regions(X, y, classifier, resolution=0.01):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    #print(xx1)
    #print(xx2)
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        #print('idx', idx)
        #print('cl', cl)
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

def decision_measure(prediction, label, method='for_if'):

  num_tp = 0
  num_tn = 0
  num_fp = 0
  num_fn = 0

  if method == 'for_if':
    for i in range(label.shape[0]):
      if label[i] == 1:
        if prediction[i] == 1:
          num_tp = num_tp + 1
        if prediction[i] == -1:
          num_fn = num_fn + 1
      if label[i] == -1:
        if prediction[i] == 1:
          num_fp = num_fp + 1
        if prediction[i] == -1:
          num_tn = num_tn + 1
  
  if method == 'mat_operat':
      comparison = np.concatenate((label.reshape(label.shape[0], 1), prediction.reshape(label.shape[0], 1)), axis =1)
      variety = np.array([[1], [2]])
      result = np.dot(comparison, variety)
      num_tp = label.shape[0] - np.count_nonzero(result-3)
      num_tn = label.shape[0] - np.count_nonzero(result+3)
      num_fp = label.shape[0] - np.count_nonzero(result-1)
      num_fn = label.shape[0] - np.count_nonzero(result+1)

  acc = (num_tp+num_tn)/(num_tp+num_tn+num_fp+num_fn)
  # pre = num_tp/(num_tp+num_fp)
  # ...
  return num_tp, num_tn, num_fp, num_fn, acc

#########################################################################
# Reading-in the Iris data
df = pd.read_csv('iris.csv', header=0, 
                 dtype={"sepal_length":"float64", 
                       "sepal_width":"float64",
                       "petal_length":"float64",
                       "petal_width":"float64",})

# Using all features
sum_tp = []
sum_fp = []
sum_fn = []
flower = ["setosa", "versicolor", "virginica"]

for f in flower:
    # Select all
    y = df.iloc[0:150, 4].values
    y = np.where(y == f, 1, -1)    
    X = df.iloc[0:150, [0, 1, 2, 3]].values
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada = AdalineSGD(n_iter=30, eta=0.01, random_state=1)
    ada.fit(X_std, y)
    y_hat = ada.predict(X_std)

    tp, tn, fp, fn, acc = decision_measure(y_hat, y, method='mat_operat')
    sum_tp.append(tp)
    sum_fp.append(fp)
    sum_fn.append(fn)

print('using all features:')
precision = sum(sum_tp)/(sum(sum_tp)+sum(sum_fp))
print('precision=', precision)
recall = sum(sum_tp)/(sum(sum_tp)+sum(sum_fn))
print('recall=', recall)
f1_score = 2*(precision*recall)/(precision+recall)
print('f1_score=', f1_score)


# label 3 class
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(["setosa", "versicolor", "virginica"]);
y_str = df.iloc[0:150, 4].values
y = le.transform(y_str)
print(y)
X = df.iloc[0:150, [0, 1, 2, 3]].values
# Choose any 2 feature
p_dic_1 = {'x1':'sepal_length','x2':'sepal_width', 'r1':0, 'r2':1}
p_dic_2 = {'x1':'sepal_length','x2':'petal_length', 'r1':0, 'r2':2}
p_dic_3 = {'x1':'sepal_length','x2':'petal_width', 'r1':0, 'r2':3}
p_dic_4 = {'x1':'sepal_width','x2':'petal_length', 'r1':1, 'r2':2}
p_dic_5 = {'x1':'sepal_width','x2':'petal_width', 'r1':1, 'r2':3}
p_dic_6 = {'x1':'petal_length','x2':'petal_width', 'r1':2, 'r2':3}
plot_type_dic = [p_dic_1, p_dic_2, p_dic_3, p_dic_4, p_dic_5, p_dic_6]
# Visualization
for plot_type in plot_type_dic:    
    # Plotting the Iris data
    plt.scatter(X[:50, plot_type['r1']], X[:50, plot_type['r2']],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, plot_type['r1']], X[50:100, plot_type['r2']],
                color='blue', marker='x', label='versicolor')
    plt.scatter(X[100:150, plot_type['r1']], X[100:150, plot_type['r2']],
                color='green', marker='^', label='virginica')

    plt.xlabel(plot_type['x1'])
    plt.ylabel(plot_type['x2'])
    plt.legend(loc='upper left')
    plt.show()

for plot_type in plot_type_dic: 
  sum_tp = []
  sum_fp = []
  sum_fn = []
  flower = ["setosa", "versicolor", "virginica"]
  for f in flower:
      
      y = df.iloc[0:150, 4].values
      y = np.where(y == f, 1, -1)    
      X = df.iloc[0:150, [plot_type['r1'], plot_type['r2']]].values
      
      X_std = np.copy(X)
      X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
      X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

      ada = AdalineSGD(n_iter=30, eta=0.01, random_state=1)
      ada.fit(X_std, y)
      y_hat = ada.predict(X_std)

      tp, tn, fp, fn, acc = decision_measure(y_hat, y, method='mat_operat')
      sum_tp.append(tp)
      sum_fp.append(fp)
      sum_fn.append(fn)

      plot_decision_regions(X_std, y, classifier=ada)
      plt.title('Adaline - Stochastic Gradient Descent')
      plt.xlabel(plot_type['x1'])
      plt.ylabel(plot_type['x2'])
      plt.legend(loc='upper left')
      plt.tight_layout()
      plt.show()

  print(plot_type['x1'], plot_type['x2'], ':')
  precision = sum(sum_tp)/(sum(sum_tp)+sum(sum_fp))
  print('precision=', precision)
  recall = sum(sum_tp)/(sum(sum_tp)+sum(sum_fn))
  print('recall=', recall)
  f1_score = 2*(precision*recall)/(precision+recall)
  print('f1_score=', f1_score)


# Correlation Matrix
feature_name = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
cm = np.corrcoef(df.iloc[:,[0, 1, 2, 3]].values.T)
# print(cm)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
yticklabels=feature_name, xticklabels=feature_name, cmap="Blues")
plt.tight_layout()
plt.show()