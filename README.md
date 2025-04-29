# Comparison of memory capacities of LSTM, GILR, and Attention models.

First, we create a data set by solving the one dimensional heat equation using Crank Nicolson on a fine mesh, then taking "measurements" on a coarse mesh. We use Dirichlet boundary conditions, setting them to zero, and randomly sample the initial conditions from the uniform distribution on the unit cube. Then, we train LSTM, GILR, and Attention models of various sizes on the data set with various batch sizes. We use the mean squared error loss and 10,000 epoch of Adam. The LSTM model is a one layer LSTM with a two layer FNN output layer. The GILR is a simplification of the LSTM model using the GILR cell of arXiv:1709.04057. The Attention model consists of one head of self-attention followed by a two layer FNN. In all three models, we vary the hidden dimension of the FNN. After running an experiment, we plot a heat map of the final training loss as a function of number of parameters and number of data points. GILR performs similarly to LSTM despite being simpler. Attention performs worse than both, suggesting that perhaps RNNs are better for time series prediction.

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
