import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Training should be done for 'epochsw' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses

        # ========================= EDIT HERE ========================
        np.random.seed() #For random seed.
        lst = np.zeros((x.shape[0],1)) #For random election
        y = y.reshape(x.shape[0],1) #Make y two dimension array.
        for epoch in range(epochs):
            for j in range(x.shape[0]): #Random Election Part
                lst[j] = j
            np.random.shuffle(lst)
            temp_x = np.zeros((batch_size, x.shape[1]))
            temp_y = np.zeros((batch_size, 1))
            for i in range(int(x.shape[0]/batch_size)):
                t = i*10
                for j in range(batch_size):
                    temp_x[j] = x[int(lst[t+j])]
                    temp_y[j] = y[int(lst[t+j])]
                y_expected = np.dot(temp_x, self.W) 
                grad = (-2.0/batch_size)*np.dot(np.transpose(temp_x),(temp_y-y_expected))
                self.W = optim.update(self.W, grad, lr)
            start = int((x.shape[0]/batch_size))*batch_size #If x.shape[0] mod batch_size != 0
            finish = x.shape[0]				    #Do last iteration
            remain=finish-start
            if remain!=0:
                temp_x = np.zeros((remain, x.shape[1]))
                temp_y = np.zeros((remain, 1))
                for j in range(remain):
                    temp_x[j] = x[int(lst[start+j])]
                    temp_y[j] = y[int(lst[start+j])]
                y_expected = np.dot(temp_x, self.W)
                grad = (-2.0/remain)*np.dot(np.transpose(temp_x),(temp_y-y_expected))
                self.W = optim.update(self.W, grad, lr)
        y = y.reshape(-1)
        tmp = y-self.eval(x).reshape(-1)
        final_loss=np.dot(tmp,tmp)/x.shape[0] #Calculate final_loss after all iter finished.
        # ============================================================
        return final_loss

    def eval(self, x):
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================
        pred = np.dot(x, self.W)
        # ============================================================
        return pred
