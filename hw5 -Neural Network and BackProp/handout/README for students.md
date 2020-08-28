# About Tiny Dataset

The tiny dataset is meant to help you debug your code using an almost minimal example. The datasets `tinyTrain.csv` and `tinyTest.csv` are of similar format to small/medium/large dataset, except that they only have 2 samples with 5 features. The corresponding detailed output `tinyOutput.txt` contains the results one would expect from a correctly implemented neural network, with the following command line arguments:

```
python3 neuralnet.py tinyTrain.csv tinyTest.csv _ _ _ 1 4 2 0.1
```

The terms that appear in the reference output for the neural network model are as follows:

```
Input: x
First layer output (i.e. hidden layer input): W_1 * x
Hidden layer output (i.e. second layer input): sigmoid(W_1 * x)
Second layer output (i.e. softmax inputs): W_2 * sigmoid(W_1 * x)
Final output: softmax(W_2 * sigmoid(W_1 * x))
```

Note that this is *only one possible* output from a correct implementation. It is perfectly fine if yours are different from this output in some aspect, e.g. different places to put the bias terms, transpose of the weight matrices, etc., as long as you can verify your correctness.