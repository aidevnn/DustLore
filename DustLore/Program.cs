﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using DustLore.Layers;
using DustLore.Losses;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore
{
    class MainClass
    {
        static void Test1()
        {
            int[] x = { 3, 6, 2, 5, 4 };
            var st = Utils.Shape2Strides(x);
            var ind0 = new int[x.Length];
            var ind1 = new int[x.Length];
            int nb = Utils.ArrMul(x);

            Console.WriteLine($"#### ({nb,3} {st.Glue(fmt: "{0,3}")})");
            Console.WriteLine();
            for (int k = 0; k < x.Length; ++k)
            {
                var y = x.ToArray();
                y[k] = 1;
                var st0 = Utils.Shape2Strides(y);
                int m = Utils.ArrMul(x, start: k);
                int n = Utils.ArrMul(x, start: k + 1);
                Console.WriteLine($"#### ({m,3} {n,3})");
                for (int i = 0; i < nb; ++i)
                {
                    Utils.Int2ArrayIndex(i, x, ind0);
                    ind0.CopyTo(ind1, 0);
                    ind1[k] = 0;

                    int j = Utils.Array2IntIndex(ind1, x, st);
                    int j0 = (i / m) * m + i % n;

                    int l = Utils.Array2IntIndex(ind1, y, st0);
                    int l0 = (i / m) * n + i % n;

                    if (j != j0 || l != l0)
                        Console.WriteLine($"i:{i,3} ({ind0.Glue()}) => ({ind1.Glue()}) j:{j,3} {j0,3} l:{l,3} {l0,3}");
                }
                Console.WriteLine();
            }
        }

        static void Test2()
        {

            int[] x = { 3, 2, 4 };
            var st = Utils.Shape2Strides(x);
            var ind0 = new int[x.Length];
            var ind1 = new int[x.Length];
            int nb = Utils.ArrMul(x);

            //Console.WriteLine($"#### ({nb,3} {st.Glue(fmt: "{0,3}")})");
            Console.WriteLine();
            for (int k = 0; k < x.Length; ++k)
            {
                var y = x.ToArray();
                y[k] = 1;
                var st0 = Utils.Shape2Strides(y);
                //int m = Utils.ArrMul(x, start: k);
                //int n = Utils.ArrMul(x, start: k + 1);
                //Console.WriteLine($"#### ({m,3} {n,3})");
                for (int i = 0; i < nb; ++i)
                {
                    Utils.Int2ArrayIndex(i, x, ind0);
                    ind0.CopyTo(ind1, 0);
                    ind1[k] = 0;

                    int j = Utils.Array2IntIndex(ind1, x, st);
                    //int j0 = (i / m) * m + i % n;

                    int l = Utils.Array2IntIndex(ind1, y, st0);
                    //int l0 = (i / m) * n + i % n;

                    //if (j != j0 || l != l0)
                    Console.WriteLine($"i:{i,3} ({ind0.Glue()}) => ({ind1.Glue()}) j:{j,3} l:{l,3} ");
                }
                Console.WriteLine();
            }
        }

        static void Test3(params int[] x)
        {
            var st = Utils.Shape2Strides(x);
            var ind0 = new int[x.Length];
            var ind1 = new int[x.Length];
            int nb = Utils.ArrMul(x);

            Console.WriteLine();
            for (int k = 0; k < x.Length; ++k)
            {
                var y = x.ToArray();
                for (int k0 = k; k0 < x.Length; ++k0)
                    y[k0] = 1;

                var st0 = Utils.Shape2Strides(y);
                var m = Utils.ArrMul(x, k);
                Console.WriteLine($"({y.Glue()}) {m}");
                for (int i = 0; i < nb; ++i)
                {
                    Utils.Int2ArrayIndex(i, x, ind0);
                    ind0.CopyTo(ind1, 0);
                    for (int k0 = k; k0 < x.Length; ++k0)
                        ind1[k0] = 0;

                    int j = Utils.Array2IntIndex(ind1, x, st0);
                    int j0 = i / m;
                    Console.WriteLine($"i:{i,3} ({ind0.Glue()}) => ({ind1.Glue()}) j:{j,3} {j0}");
                }
                Console.WriteLine();
            }
        }

        static void Test4(params int[] x)
        {
            var st = Utils.Shape2Strides(x);
            var ind0 = new int[x.Length];
            var ind1 = new int[x.Length];
            int nb = Utils.ArrMul(x);

            Console.WriteLine();
            for (int k = 0; k < x.Length; ++k)
            {
                var y = x.ToArray();
                for (int k0 = 0; k0 < k; ++k0)
                    y[k0] = 1;

                var st0 = Utils.Shape2Strides(y);
                var m = Utils.ArrMul(x, k);
                Console.WriteLine($"({y.Glue()}) {m}");
                for (int i = 0; i < nb; ++i)
                {
                    Utils.Int2ArrayIndex(i, x, ind0);
                    ind0.CopyTo(ind1, 0);
                    for (int k0 = 0; k0 < k; ++k0)
                        ind1[k0] = 0;

                    int j = Utils.Array2IntIndex(ind1, x, st0);
                    int j0 = i % m;
                    Console.WriteLine($"i:{i,3} ({ind0.Glue()}) => ({ind1.Glue()}) j:{j,3} {j0}");
                }
                Console.WriteLine();
            }
        }

        static void Test5()
        {
            void testBroadcast(int[] s0, int[] s1)
            {
                var p = Utils.BroadCastShapes(s0, s1);
                Console.WriteLine($"({s0.Glue()}) ({s1.Glue()}) => {p.Item1} {p.Item2} ({p.Item3.Glue()})");
            }

            testBroadcast(new int[] { 2, 3 }, new int[] { 2, 3 });
            testBroadcast(new int[] { 3 }, new int[] { 2, 3 });
            testBroadcast(new int[] { 2, 1 }, new int[] { 2, 3 });
            testBroadcast(new int[] { 2, 3 }, new int[] { 3 });
            testBroadcast(new int[] { 2, 3 }, new int[] { 2, 1 });
            testBroadcast(new int[] { 1, 3 }, new int[] { 2, 1 });
            testBroadcast(new int[] { 3 }, new int[] { 2, 1 });
            testBroadcast(new int[] { 2, 1 }, new int[] { 1, 3 });
            testBroadcast(new int[] { 2, 1 }, new int[] { 3 });
            testBroadcast(new int[] { 4, 2, 3 }, new int[] { 2, 3 });
            testBroadcast(new int[] { 4, 2, 1 }, new int[] { 2, 3 });
            testBroadcast(new int[] { 4, 2, 1 }, new int[] { 3 });
            testBroadcast(new int[] { 4, 1, 1 }, new int[] { 3 });
            testBroadcast(new int[] { 2, 3 }, new int[] { 1 });
        }

        static void Test6()
        {
            var x = ND.Uniform(0, 10, 4, 2, 3);
            Console.WriteLine(x);
            Console.WriteLine($"a={x}");
            for (int k = 0; k < x.Shape.Length; ++k)
            {
                Console.WriteLine($"a.sum(axis={k})");
                Console.WriteLine(ND.SumAxis(x, k));
                //Console.WriteLine($"a.sum(axis={k}, keepdims=True)");
                //Console.WriteLine(x.SumAxis(k, true));
            }
        }

        static void Test7()
        {

            var x0 = ND.Uniform(0, 10, 3, 4).Cast<double>();
            var x1 = ND.Uniform(0, 10, 5, 3).Cast<double>();
            var c = ND.Uniform(0, 10, 1, 5).Cast<double>();
            Console.WriteLine(x0);
            Console.WriteLine(x1);
            Console.WriteLine(c);
            Console.WriteLine($"a={x0}");
            Console.WriteLine($"b={x1}");
            Console.WriteLine($"c={c}");
            Console.WriteLine("np.dot(a.T,b.T)+c");
            Console.WriteLine(NDsharp.GemmTATBC(x0, x1, c));
            Console.WriteLine(NDmkl.GemmTATBC(x0, x1, c));
        }

        static void Test8()
        {
            int N = 5;
            double[][] y0 = new double[N][];
            double[][] p0 = new double[N][];
            for (int k = 0; k < N; ++k)
            {
                var y = Enumerable.Range(0, 4).Select(Convert.ToDouble).OrderBy(x => Utils.Random.NextDouble()).ToArray();
                var p = Enumerable.Range(0, 4).Select(Convert.ToDouble).OrderBy(x => Utils.Random.NextDouble()).ToArray();
                y0[k] = y;
                p0[k] = p;
            }

            var Y = new NDarray<double>(y0).Reshape(-1, 2, 2);
            var P = new NDarray<double>(p0).Reshape(-1, 2, 2);

            var adm = new Adam();
            var b = new BatchNormalizeLayer() { IsTraining = true };
            b.SetInputShape(new int[] { 2, 2 });
            b.Initialize(adm);

            Console.WriteLine(Y);
            Console.WriteLine(P);

            Console.WriteLine(b.Forward(Y, true));
            Console.WriteLine(b.Backward(P));
        }

        static void Test9()
        {
            var mp = new MaxPool2dLayer((2, 2));
            mp.SetInputShape(new int[] { 1, 8, 8 });
            var imgs = ND.Uniform(0, 10, 1, 1, 8, 8).Cast<double>();
            var imgs0 = mp.Forward(imgs, true);
            var imgs1 = mp.Backward(imgs0);
            Console.WriteLine(imgs);
            Console.WriteLine(imgs0);
            Console.WriteLine(imgs1);

            //var imgs = ND.Uniform(0, 10, 1, 1, 8, 8).Cast<double>();
            var xcols = Images2Columns.Images2Columns2Dfast(imgs, (2, 2), 2, "valid");
            var mx = ND.MaxAxis(xcols, 0).Reshape(4, 4, 1, 1).Transpose(2, 3, 0, 1);
            Console.WriteLine(imgs);
            Console.WriteLine(mx);
        }

        static void Test10()
        {
            string fmt = "{0,3}";
            int[] x = { 3, 5, 2, 4 };
            for (int axis = 0; axis < x.Length; ++axis)
            {

                int[] y = Utils.PrepareAxisOps(x, axis, true);
                var ind = new int[y.Length];
                int nb = Utils.ArrMul(y);

                var st = Utils.Shape2Strides(x);
                int m = Utils.ArrMul(x);
                var n = st[axis];
                int o = x[axis];

                Console.WriteLine($"### Shape:({x.Glue()}) Axis:{axis} Step:{n}");
                for (int i = 0; i < nb; ++i)
                {
                    Utils.Int2ArrayIndex(i, y, ind);
                    List<int> lt = new List<int>();
                    for (int j = 0; j < o; ++j)
                    {
                        ind[axis] = j;
                        int i1 = Utils.Array2IntIndex(ind, x, st);
                        lt.Add(i1);
                    }

                    int k = (i / n) * n * o + (i % n);
                    Console.WriteLine($"{i,3} => ({k,3}) => ({lt.Glue(fmt: fmt)})");
                }

                Console.WriteLine();
            }
        }

        static void Test11(params int[] shape)
        {
            var x = ND.Uniform(0, 10, shape).Cast<double>();
            Console.WriteLine($"x={x}");
            for (int axis = 0; axis < shape.Length; ++axis)
            {
                var m = ND.ArgmaxAxis(x, axis);
                Console.WriteLine($"np.argmax(x, axis={axis})");
                Console.WriteLine(m);
            }
        }

        static void TestXor(bool summary = false, int epochs = 50, int displayEpochs = 25)
        {
            Console.WriteLine("Hello World, MLP on Xor Dataset.");

            double[,] X0 = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } };
            double[,] y0 = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };
            double[][] X = Enumerable.Range(0, 4).Select(i => Enumerable.Range(0, 2).Select(j => X0[i, j]).ToArray()).ToArray();
            double[][] y = Enumerable.Range(0, 4).Select(i => Enumerable.Range(0, 1).Select(j => y0[i, j]).ToArray()).ToArray();
            var ndX = new NDarray<double>(X);
            var ndY = new NDarray<double>(y);

            var net = new Network(new SGD(0.2, 0.2), new MeanSquaredLoss(), new RoundAccuracy());
            net.AddLayer(new DenseLayer(8, inputShape: 2));
            net.AddLayer(new TanhLayer());
            net.AddLayer(new DenseLayer(1));
            net.AddLayer(new SigmoidLayer());

            if (summary)
                net.Summary();

            net.Fit(ndX, ndY, epochs, displayEpochs: displayEpochs);

            if (summary)
            {
                var yp = net.Forward(ndX);
                for (int k = 0; k < 4; ++k)
                    Console.WriteLine($"[{X[k].Glue()}] = [{y[k][0]}] -> {yp.Data[k]:0.000000}");
            }

            Console.WriteLine();
        }

        static void TestIris(bool summary = false, int epochs = 50, int displayEpochs = 25, int batchsize = 10)
        {
            Console.WriteLine("Hello World, MLP on Iris Dataset.");

            (var trainX, var trainY, var testX, var testY) = ImportDataset.IrisDataset(ratio: 0.8);
            var net = new Network(new SGD(0.025, 0.2), new MeanSquaredLoss(), new ArgmaxAccuracy());
            net.AddLayer(new DenseLayer(5, inputShape: 4));
            net.AddLayer(new TanhLayer());
            net.AddLayer(new DenseLayer(3));
            net.AddLayer(new SigmoidLayer());

            if (summary)
                net.Summary();

            net.Fit(trainX, trainY, epochs, batchSize: batchsize, displayEpochs: displayEpochs);
            net.Test(testX, testY);

            Console.WriteLine();
        }

        static void TestDigits(bool summary = false, int epochs = 50, int displayEpochs = 25, int batchsize = 100)
        {
            Console.WriteLine("Hello World, MLP on Digits Dataset.");

            (var trainX, var trainY, var testX, var testY) = ImportDataset.DigitsDataset(ratio: 0.9);
            var net = new Network(new SGD(0.02), new CrossEntropyLoss(), new ArgmaxAccuracy());
            net.AddLayer(new DenseLayer(32, inputShape: 64));
            net.AddLayer(new SigmoidLayer());
            net.AddLayer(new DenseLayer(10));
            net.AddLayer(new SoftmaxLayer());

            if (summary)
                net.Summary();

            net.Fit(trainX, trainY, testX, testY, epochs: epochs, batchSize: 100, displayEpochs: displayEpochs);
            //net.Test(testX, testY);

            Console.WriteLine();
        }

        static void TestDigitsCNN(int epochs = 5, int displayEpochs = 1)
        {
            ND.Backend = Backend.MKL;
            Console.WriteLine($"Hello World, CNN on Digits Dataset. Backend {ND.Backend}");

            (var trainX, var trainY, var testX, var testY) = ImportDataset.DigitsDataset(ratio: 0.9, normalize: false);
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

            net.Summary();

            net.Fit(trainX, trainY, testX, testY, epochs: epochs, batchSize: 64, displayEpochs: displayEpochs);
        }

        static void TestDigitsCNN2(int epochs = 5, int displayEpochs = 1)
        {
            ND.Backend = Backend.MKL;
            Console.WriteLine($"Hello World, CNN on Digits Dataset. Backend {ND.Backend}");

            (var trainX, var trainY, var testX, var testY) = ImportDataset.DigitsDataset(ratio: 0.6, normalize: false);
            trainX.ReshapeInplace(-1, 1, 8, 8);
            testX.ReshapeInplace(-1, 1, 8, 8);

            var net = new Network(new Adam(), new CrossEntropyLoss(), new ArgmaxAccuracy());

            net.AddLayer(new Conv2dLayer(nfilters: 16, filterShape: (3, 3), inputShape: (1, 8, 8), padding: "same", strides: 1));
            net.AddLayer(new ReluLayer());
            net.AddLayer(new Conv2dLayer(nfilters: 32, filterShape: (3, 3), padding: "same", strides: 1));
            net.AddLayer(new ReluLayer());
            net.AddLayer(new MaxPooling2dLayer((2, 2)));
            net.AddLayer(new BatchNormalizeLayer());
            net.AddLayer(new DropoutLayer(0.25));
            net.AddLayer(new FlattenLayer());
            net.AddLayer(new DenseLayer(256));
            net.AddLayer(new ReluLayer());
            net.AddLayer(new DropoutLayer(0.4));
            net.AddLayer(new DenseLayer(32));
            net.AddLayer(new ReluLayer());
            net.AddLayer(new DenseLayer(10));
            net.AddLayer(new SoftmaxLayer());

            net.Summary();

            net.Fit(trainX, trainY, testX, testY, epochs: epochs, batchSize: 64, displayEpochs: displayEpochs);
        }

        static void TestRNN()
        {
            ND.Backend = Backend.MKL;
            Console.WriteLine($"Hello World, RNN on Sequence Dataset. Backend {ND.Backend}");

            (var trainX, var trainY, var testX, var testY) = ImportDataset.SequenceDataset(250, 0.8);
            var net = new Network(new Adam(), new CrossEntropyLoss(), new ArgmaxAccuracy());

            net.AddLayer(new RnnLayer(nUnits: 10, inputShape: (10, 61)));
            net.AddLayer(new SoftmaxLayer());

            net.Summary();

            net.Fit(trainX, trainY, testX, testY, 500, 20, 50);

            var pred = net.Predict(testX);
            var apred = ND.ArgmaxAxis(pred, -1);
            var atestX = ND.ArgmaxAxis(testX, -1);
            var atestY = ND.ArgmaxAxis(testY, -1);
            for (int i = 0; i < 5; ++i)
            {
                var x = atestX.GetAtIndex(i);
                var y = atestY.GetAtIndex(i);
                var p = apred.GetAtIndex(i);
                Console.WriteLine();
                Console.WriteLine("Sample Test");
                Console.WriteLine("X=[{0}]", x.Glue(fmt: "{0,2}"));
                Console.WriteLine("y=[{0}]", y.Glue(fmt: "{0,2}"));
                Console.WriteLine("p=[{0}]", p.Glue(fmt: "{0,2}"));
            }
        }

        public static void Main(string[] args)
        {
            //TestXor(true, 500, 50);
            TestIris(true, 50, 5);
            //TestDigits(true, 50, 5);

            //ND.Backend = Backend.MKL;
            //for (int k = 0; k < 5; ++k) TestIris();

            //TestDigitsCNN(50, 1);
            //TestDigitsCNN2(50, 1);

            //TestRNN();

        }
    }
}
