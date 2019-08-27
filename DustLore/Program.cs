using System;
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

            net.Fit(ndX, ndY, epochs, displayEpochs);

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

            net.Fit(trainX, trainY, epochs, displayEpochs, batchsize);
            net.Test(testX, testY);

            Console.WriteLine();
        }

        static void TestDigits(bool summary = false, int epochs = 50, int displayEpochs = 25, int batchsize = 100)
        {
            Console.WriteLine("Hello World, MLP on Digits Dataset.");

            (var trainX, var trainY, var testX, var testY) = ImportDataset.DigitsDataset(ratio: 0.9);
            var net = new Network(new SGD(0.025, 0.2), new CrossEntropyLoss(), new ArgmaxAccuracy());
            net.AddLayer(new DenseLayer(32, inputShape: 64));
            net.AddLayer(new SigmoidLayer());
            net.AddLayer(new DenseLayer(10));
            net.AddLayer(new SigmoidLayer());

            if (summary)
                net.Summary();

            net.Fit(trainX, trainY, epochs, displayEpochs, batchsize);
            net.Test(testX, testY);

            Console.WriteLine();
        }

        static void TestDigitsCNN(int epochs = 5, int displayEpochs = 1)
        {
            ND.Backend = Backend.MKL;
            Console.WriteLine($"Hello World, CNN on Digits Dataset. Backend {ND.Backend}");

            (var trainX, var trainY, var testX, var testY) = ImportDataset.DigitsDataset(ratio: 0.9);
            trainX.ReshapeInplace(-1, 1, 8, 8);
            testX.ReshapeInplace(-1, 1, 8, 8);

            var net = new Network(new Adam(), new CrossEntropyLoss(), new ArgmaxAccuracy());

            net.AddLayer(new Conv2d(nfilters: 16, filterShape: (3, 3), inputShape: (1, 8, 8), padding: "same", strides: 1));
            net.AddLayer(new ReluLayer());
            net.AddLayer(new DropoutLayer(0.25));
            net.AddLayer(new BatchNormalizeLayer());
            net.AddLayer(new Conv2d(nfilters: 32, filterShape: (3, 3), padding: "same", strides: 1));
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

            net.Fit(trainX, trainY, testX, testY, epochs: epochs, batchSize: 100, displayEpochs: displayEpochs);
        }

        public static void Main(string[] args)
        {
            //TestXor(true, 500, 50);
            //TestIris(true, 50, 5);
            //TestDigits(true, 50, 5);

            //for (int k = 0; k < 5; ++k) TestDigits();

            TestDigitsCNN(50, 1);
        }
    }
}
