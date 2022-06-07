from numpy import * # i am lazy to specify

# lets make a class for the math stuff (I dont want to fiddle with it much) and althouhg I can instruct the algorithm to just take the seconed number in array lets make the machine learn it

class nn():
    def __init__(self):
        random.seed(1)
        # for the 3 x 1 matrix we want to fill it with value in range -1 to 1
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return (1 / (1 + exp(-x))) # read the readme for understanding what is this

    def __sigmoid_derivative(self, x):
        return x * (1-x)
    # tsis = training_set_inputs
    # tsos = training_set_outputs
    # n = number of trials 

    def train(self, tsis, tsos, n):
        for i in range(n):
            o = self.think(tsis)
            e = tsos - o
            a = dot(tsis.T, e * self.__sigmoid_derivative(o))
            self.synaptic_weights += a

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__=="__main__":
    nn = nn() # initalize a new neural network

    print("Random starting synaptic weight: ")
    print(nn.synaptic_weights)

    training_set_input = array([[0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]])
    training_set_output = array([[0, 1, 1, 0, 1]]).T

    nn.train(training_set_input, training_set_output, 10000)

    print("New synaptic_weights: ")
    print(nn.synaptic_weights)

    print("Under New situation [1, 0, 1]: ")
    print(nn.think(array([1, 0, 1])))
