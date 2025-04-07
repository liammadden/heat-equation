# Learning the 1D Heat Equation with a LSTM Model

We create training and test data sets by solving the one dimensional heat equation using Crank Nicolson on a fine mesh, then taking "measurements" on a coarse mesh. We use Dirichlet boundary conditions, setting them to zero, and randomly sample the initial conditions from the uniform distribution on the unit cube. Our parameterized model is a one layer LSTM with a linear output layer. Given a sequence of input vectors, the LSTM produces a sequence of hidden vectors. The output layer transforms each hidden vector to an output vector that is meant to predict the next input vector from the sequence. We use mean squared error to quantify the loss between the predicted vectors and actual vectors. We train the model on the training data set using 10,000 epochs of full batch Adam, then compute the training loss and test loss. We vary the hidden dimension to see how the final training loss and final test loss depend on the number of parameters. What we find is the usual bias-variance tradeoff. The minimum test loss seems to occur at around 2700 parameters, which is three times the number of data points (the number of data points is the number of sequences times one less than the number of vectors in each sequence).

## Installing Required Packages

To install the required python packages, use the following command:

```
pip install -r requirements.txt
```

## Running the Code

To run the code, use the following command:

```
python run_experiments.py
```
