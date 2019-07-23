import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses

        # ========================= EDIT HERE ========================
#Similar with Linear Regression
        np.random.seed()
        lst = np.zeros((x.shape[0], 1))
        y = y.reshape(x.shape[0],1)
        for epoch in range(epochs):
            for j in range(x.shape[0]):
                lst[j] = j
            np.random.shuffle(lst)
            temp_x = np.zeros((batch_size, x.shape[1]))
            temp_y = np.zeros((batch_size, 1))
            for i in range(x.shape[0]//batch_size):
                t = i*batch_size
                for j in range(batch_size):
                    temp_x[j] = x[int(lst[t+j])]
                    temp_y[j] = y[int(lst[t+j])]
                y_expected = self._sigmoid(np.dot(temp_x, self.W))
                grad = (1.0/batch_size)*np.dot(np.transpose(temp_x), y_expected-temp_y)
                self.W = optim.update(self.W, grad, lr)
            start = (x.shape[0]//batch_size)*batch_size
            finish = x.shape[0]
            remain = finish-start
            if remain != 0 :
                temp_x = np.zeros((remain, x.shape[1]))
                temp_y = np.zeros((remain, 1))
                for j in range(remain):
                    temp_x[j] = x[int(lst[start+j])]
                    temp_y[j] = y[int(lst[start+j])]
                y_expected = self._sigmoid(np.dot(temp_x, self.W))
                grad = (1.0/remain)*np.dot(np.transpose(temp_x), y_expected-temp_y)
                self.W = optim.update(self.W, grad, lr)
        h = np.dot(x, self.W)
        final_loss = -(1.0/x.shape[0])*np.sum(h*(y-1.0) - np.log(1.0+np.exp(-h)))
        # ============================================================
        return final_loss

    def eval(self, x):
        threshold = 0.5
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        last = self._sigmoid(np.dot(x, self.W))
        pred = (last >= threshold).astype(int)
        # ============================================================

        return pred

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        sigmoid = 1 / (1+np.exp(-x))
        if np.any(sigmoid == 1.0):
            sigmoid = sigmoid - 1e-16
        if np.any(sigmoid == 0.0):
            sigmoid = sigmoid + 1e-16
        # ============================================================
        return sigmoid
