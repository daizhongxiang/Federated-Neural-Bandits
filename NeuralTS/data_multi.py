from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np

from sklearn.preprocessing import StandardScaler # added by us
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder # added by us

np.random.seed(0)


class Bandit_multi:
    def __init__(self, name, is_shuffle=True, seed=None):
        # Fetch data
        if name == 'mnist':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
            
#             scaler = StandardScaler()
#             X = scaler.fit_transform(X)

        elif name == 'mushroom':
            X, y = fetch_openml('mushroom', version=1, return_X_y=True)
            
            X = np.array(X)
            y = np.array(y)

            enc = OrdinalEncoder()
#             enc = OneHotEncoder(handle_unknown='ignore')

            X = enc.fit_transform(X)
#             X = X.todense()

#             avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        elif name == 'adult':
            X, y = fetch_openml('adult', version=2, return_X_y=True)
            
            X = np.array(X)
            y = np.array(y)

            enc = OrdinalEncoder()
#             enc = OneHotEncoder(handle_unknown='ignore')

            X = enc.fit_transform(X)
            
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'covertype':
            X, y = fetch_openml('covertype', version=3, return_X_y=True)
            
            X = np.array(X).astype(float)
            y = np.array(y)
            
            le = LabelEncoder()
            y = le.fit_transform(y)

            
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'isolet':
            X, y = fetch_openml('isolet', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'letter':
            X, y = fetch_openml('letter', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'MagicTelescope':
            X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
            
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        elif name == 'shuttle':
            X, y = fetch_openml('shuttle', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
            
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        else:
            raise RuntimeError('Dataset does not exist')
        # Shuffle data
        if is_shuffle:
            self.X, self.y = shuffle(X, y, random_state=seed)
        else:
            self.X, self.y = X, y
        
        self.y = np.array(self.y)
        
        # generate one_hot coding:
        self.y_arm = OrdinalEncoder(
            dtype=np.int).fit_transform(self.y.reshape((-1, 1)))
        # cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = np.max(self.y_arm) + 1
        self.dim = self.X.shape[1] * self.n_arm
        self.act_dim = self.X.shape[1]

    def step(self):
        assert self.cursor < self.size
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                self.act_dim] = self.X[self.cursor]
        arm = self.y_arm[self.cursor][0]
        rwd = np.zeros((self.n_arm,))
        rwd[arm] = 1
        self.cursor += 1
        return X, rwd

    def finish(self):
        return self.cursor == self.size

    def reset(self):
        self.cursor = 0


if __name__ == '__main__':
    b = Bandit_multi('mushroom')
    x_y, a = b.step()
    # print(x_y[0])
    # print(x_y[1])
    # print(np.linalg.norm(x_y[0]))
