import numpy as np


class Converting:
    def __init__(self):

        # input dataset
        self.training_inputs = np.array([[0,0,1],
                                    [1,1,1],
                                    [1,0,1],
                                    [0,1,1]])
        # output dataset
        self.training_outputs = np.array([[0.0,1.0,1.0,0]]).T

        np.random.seed(1)
        self.random_weights = 2 * np.random.random((3, 1)) - 1
        print("random starting weights")
        print(self.random_weights)

    def sigmoid(self,x):
        return 1 / ( 1 + np.exp(-x) )

    def check(self):
        output = np.dot(self.training_inputs , self.random_weights)
        self.activiation_output = self.sigmoid(output)
       
        print(self.activiation_output)

con = Converting()
con.check()
