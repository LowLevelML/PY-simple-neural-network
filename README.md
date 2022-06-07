# PY-simple-neural-network

This is a simple neural network written in python (check the [src](./src/) directory).

# Explanation

So first you need an input matrix

if such a matrix is given as such:

|-     -|
| 0 0 1 |
| 1 1 1 |
| 1 0 1 |
| 0 1 1 |
| 1 1 0 |
|-     -|

the output I want is a 1 x 4 matrix (or 4 x 1 I dont really like matrixes)

|- -|
| 0 |
| 1 |
| 0 |
| 1 |
| 1 |
|- -|


now if you did not already notice the pattern I was going for, I was going for the second row as my output (but the computer does not know that yet).

## Training

I think this explanation below from source [1] sums it up perfectly

```console
But how do we teach our neuron to answer the question correctly? We will give each input a weight, which can be a positive or negative number. An input with a large positive weight or a large negative weight, will have a strong effect on the neuron’s output. Before we start, we set each weight to a random number. Then we begin the training process:

    1) Take the inputs from a training set example, adjust them by the weights, and pass them through a special formula to calculate the neuron’s output.
    2) Calculate the error, which is the difference between the neuron’s output and the desired output in the training set example.
    3) Depending on the direction of the error, adjust the weights slightly.
    4) Repeat this process n times.

```


# Sources

- [How to build a simple neural network in 9 lines of Python code](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1)

# License

`SPDX-License-Identifier: MIT AND GPL-3.0-or-later`

The following licenses are present in the licenses folder.

If you are going to use my code without my permision by forking it, you must follow both the MIT and GPL-3.0-or-later licenses.
