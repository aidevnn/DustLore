# DustLore

An example of a CNN and RNN. The Intel MKL is used for better speed, but only the matrix multiplication gemm was used at this time.
The futur challenge is may be to rewrite all code in C++ and compiling directly with Intel C++ Compiler for better performance and educative purpose without requiring the MKL library.
The ultimate goal is to use NVIDIA CUDA but it is another long and hard work. Actually, a double precision floatting number is used, but it is 2 times slower than single precision float, and it will be the next improvement.

The dataset for the CNN is digits handwrite from scikit.
The definition of the CNN network.
```
(var trainX, var trainY, var testX, var testY) = ImportDataset.DigitsDataset(ratio: 0.9);
trainX.ReshapeInplace(-1, 1, 8, 8);
testX.ReshapeInplace(-1, 1, 8, 8);

var net = new Network(new Adam(), new CrossEntropyLoss(), new ArgmaxAccuracy());

net.AddLayer(new Conv2dLayer(nfilters: 16, filterShape: (3, 3), inputShape: (1, 8, 8), padding: "same", strides: 1));
net.AddLayer(new ReluLayer());
net.AddLayer(new DropoutLayer(0.25));
net.AddLayer(new BatchNormalizeLayer());
net.AddLayer(new Conv2dLayer(nfilters: 32, filterShape: (3, 3), padding: "same", strides: 1));
net.AddLayer(new ReluLayer());
net.AddLayer(new DropoutLayer(0.25));
net.AddLayer(new BatchNormalizeLayer());
net.AddLayer(new FlattenLayer());
net.AddLayer(new DenseLayer(256));
net.AddLayer(new ReluLayer());
net.AddLayer(new DropoutLayer(0.4));
net.AddLayer(new BatchNormalizeLayer());
net.AddLayer(new DenseLayer(10));
net.AddLayer(new SoftmaxLayer());

net.Summary(true);

net.Fit(trainX, trainY, testX, testY, epochs: 50, batchSize: 100, displayEpochs: 1);
```

The output of the CNN network.
```
Hello World, CNN on Digits Dataset. Backend MKL
Train on 1617 / Test on 180
Summary
Network: Adam / CrossEntropyLoss / ArgmaxAccuracy
Input  Shape:(1 8 8)
Layer: Conv2dLayer          Parameters:     416 Nodes[In:   (1 8 8) -> Out:  (16 8 8)]
Layer: ReluLayer            Parameters:       0 Nodes[In:  (16 8 8) -> Out:  (16 8 8)]
Layer: DropoutLayer         Parameters:       0 Nodes[In:  (16 8 8) -> Out:  (16 8 8)]
Layer: BatchNormalizeLayer  Parameters:    2048 Nodes[In:  (16 8 8) -> Out:  (16 8 8)]
Layer: Conv2dLayer          Parameters:   12832 Nodes[In:  (16 8 8) -> Out:  (32 8 8)]
Layer: ReluLayer            Parameters:       0 Nodes[In:  (32 8 8) -> Out:  (32 8 8)]
Layer: DropoutLayer         Parameters:       0 Nodes[In:  (32 8 8) -> Out:  (32 8 8)]
Layer: BatchNormalizeLayer  Parameters:    4096 Nodes[In:  (32 8 8) -> Out:  (32 8 8)]
Layer: FlattenLayer         Parameters:       0 Nodes[In:  (32 8 8) -> Out:    (2048)]
Layer: DenseLayer           Parameters:  524544 Nodes[In:    (2048) -> Out:     (256)]
Layer: ReluLayer            Parameters:       0 Nodes[In:     (256) -> Out:     (256)]
Layer: DropoutLayer         Parameters:       0 Nodes[In:     (256) -> Out:     (256)]
Layer: BatchNormalizeLayer  Parameters:     512 Nodes[In:     (256) -> Out:     (256)]
Layer: DenseLayer           Parameters:    2570 Nodes[In:     (256) -> Out:      (10)]
Layer: SoftmaxLayer         Parameters:       0 Nodes[In:      (10) -> Out:      (10)]
Output Shape:(10)
Total Parameters:547018

Epoch:    0/50. loss:0.084124 acc:0.8300; Validation. loss:0.185877 acc:0.9000 Time:      2975 ms
Epoch:    1/50. loss:0.027805 acc:0.9481; Validation. loss:0.060896 acc:0.9333 Time:      5989 ms
Epoch:    2/50. loss:0.020745 acc:0.9650; Validation. loss:0.039423 acc:0.9556 Time:      9008 ms
Epoch:    3/50. loss:0.015517 acc:0.9713; Validation. loss:0.040907 acc:0.9500 Time:     12027 ms
Epoch:    4/50. loss:0.014707 acc:0.9719; Validation. loss:0.027039 acc:0.9611 Time:     15051 ms
Epoch:    5/50. loss:0.012248 acc:0.9806; Validation. loss:0.027649 acc:0.9556 Time:     18063 ms
Epoch:    6/50. loss:0.008596 acc:0.9825; Validation. loss:0.026016 acc:0.9556 Time:     21096 ms
Epoch:    7/50. loss:0.009357 acc:0.9813; Validation. loss:0.029946 acc:0.9556 Time:     24128 ms
Epoch:    8/50. loss:0.008723 acc:0.9875; Validation. loss:0.026786 acc:0.9611 Time:     27170 ms
Epoch:    9/50. loss:0.006190 acc:0.9900; Validation. loss:0.024057 acc:0.9556 Time:     30198 ms
Epoch:   10/50. loss:0.007371 acc:0.9881; Validation. loss:0.027491 acc:0.9556 Time:     33232 ms
.........
Epoch:   40/50. loss:0.004483 acc:0.9913; Validation. loss:0.021858 acc:0.9611 Time:    125606 ms
Epoch:   41/50. loss:0.001881 acc:0.9963; Validation. loss:0.017646 acc:0.9722 Time:    128737 ms
Epoch:   42/50. loss:0.001607 acc:0.9981; Validation. loss:0.019852 acc:0.9722 Time:    131917 ms
Epoch:   43/50. loss:0.002071 acc:0.9963; Validation. loss:0.019857 acc:0.9722 Time:    135196 ms
Epoch:   44/50. loss:0.002332 acc:0.9981; Validation. loss:0.017949 acc:0.9722 Time:    138411 ms
Epoch:   45/50. loss:0.002316 acc:0.9963; Validation. loss:0.019492 acc:0.9722 Time:    141578 ms
Epoch:   46/50. loss:0.001705 acc:0.9963; Validation. loss:0.020997 acc:0.9722 Time:    144702 ms
Epoch:   47/50. loss:0.003131 acc:0.9950; Validation. loss:0.020853 acc:0.9722 Time:    147787 ms
Epoch:   48/50. loss:0.002444 acc:0.9950; Validation. loss:0.025522 acc:0.9667 Time:    150830 ms
Epoch:   49/50. loss:0.002052 acc:0.9956; Validation. loss:0.026849 acc:0.9667 Time:    153881 ms
Epoch:   50/50. loss:0.001430 acc:0.9963; Validation. loss:0.027336 acc:0.9667 Time:    156936 ms
Time:156936 ms
```

The dataset for the RNN is a sequence of integer.
The definition of the RNN network.
```
(var trainX, var trainY, var testX, var testY) = ImportDataset.SequenceDataset(250, 0.8);
var net = new Network(new Adam(), new CrossEntropyLoss(), new ArgmaxAccuracy());

net.AddLayer(new RnnLayer(nUnits: 10, inputShape: (10, 61)));
net.AddLayer(new SoftmaxLayer());

net.Summary(true);

net.Fit(trainX, trainY, testX, testY, 500, 20, 50);
```

The output of the RNN network.
```
Hello World, RNN on Sequence Dataset. Backend MKL
Train on 200 / Test on 50
Summary
Network: Adam / CrossEntropyLoss / ArgmaxAccuracy
Input  Shape:(10 61)
Layer: RnnLayer             Parameters:    1320 Nodes[In:   (10 61) -> Out:   (10 61)]
Layer: SoftmaxLayer         Parameters:       0 Nodes[In:   (10 61) -> Out:   (10 61)]
Output Shape:(10 61)
Total Parameters:1320

Epoch:    0/500. loss:0.083112 acc:0.0515; Validation. loss:0.082722 acc:0.1000 Time:       123 ms
Epoch:   50/500. loss:0.043852 acc:0.5765; Validation. loss:0.042981 acc:0.6200 Time:      3128 ms
Epoch:  100/500. loss:0.032107 acc:0.7350; Validation. loss:0.031252 acc:0.7420 Time:      6116 ms
Epoch:  150/500. loss:0.023720 acc:0.8465; Validation. loss:0.023054 acc:0.8600 Time:      9062 ms
Epoch:  200/500. loss:0.017329 acc:0.9595; Validation. loss:0.016810 acc:0.9700 Time:     11977 ms
Epoch:  250/500. loss:0.012618 acc:0.9595; Validation. loss:0.012182 acc:0.9700 Time:     14916 ms
Epoch:  300/500. loss:0.009190 acc:0.9595; Validation. loss:0.008837 acc:0.9700 Time:     17716 ms
Epoch:  350/500. loss:0.006761 acc:1.0000; Validation. loss:0.006484 acc:1.0000 Time:     20617 ms
Epoch:  400/500. loss:0.005029 acc:1.0000; Validation. loss:0.004817 acc:1.0000 Time:     23514 ms
Epoch:  450/500. loss:0.003786 acc:1.0000; Validation. loss:0.003620 acc:1.0000 Time:     26374 ms
Epoch:  500/500. loss:0.002883 acc:1.0000; Validation. loss:0.002751 acc:1.0000 Time:     29465 ms
Time:29465 ms

Sample Test
X=[ 3  6  9 12 15 18 21 24 27 30]
y=[ 6  9 12 15 18 21 24 27 30  1]
p=[ 6  9 12 15 18 21 24 27 30  1]

Sample Test
X=[ 6 12 18 24 30 36 42 48 54 60]
y=[12 18 24 30 36 42 48 54 60  1]
p=[12 18 24 30 36 42 48 54 60  1]

Sample Test
X=[ 4  8 12 16 20 24 28 32 36 40]
y=[ 8 12 16 20 24 28 32 36 40  1]
p=[ 8 12 16 20 24 28 32 36 40  1]

Sample Test
X=[ 5 10 15 20 25 30 35 40 45 50]
y=[10 15 20 25 30 35 40 45 50  1]
p=[10 15 20 25 30 35 40 45 50  1]
```

### Reference
The original code is from this repo in Python https://github.com/eriklindernoren/ML-From-Scratch