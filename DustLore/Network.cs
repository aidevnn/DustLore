using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using DustLore.Activations;
using DustLore.Layers;
using DustLore.Losses;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore
{
    public class Network
    {
        readonly IOptimizer optimizer;
        readonly ILoss loss;
        readonly IAccuracy accuracy;

        public Network(IOptimizer optimizer, ILoss loss, IAccuracy accuracy)
        {
            this.optimizer = optimizer;
            this.loss = loss;
            this.accuracy = accuracy;
        }

        List<ILayer> layers = new List<ILayer>();
        public void AddLayer(ILayer layer)
        {
            if (layers.Count != 0)
                layer.SetInputShape(layers.Last().GetOutputShape());

            layer.Initialize(optimizer);
            layers.Add(layer);
        }

        public void SetTraining(bool isTraining) => layers.ForEach(l => l.IsTraining = isTraining);

        public NDarray<double> Forward(NDarray<double> X, bool isTraining = true)
        {
            foreach (var l in layers)
                X = l.Forward(X, isTraining);

            return X;
        }

        public void Backward(NDarray<double> accumGrad)
        {
            foreach (var l in layers.Reverse<ILayer>())
                accumGrad = l.Backward(accumGrad);
        }

        public NDarray<double> Predict(NDarray<double> X) => Forward(X, false);

        public (double, double) TestOnBatch(NDarray<double> X, NDarray<double> y)
        {
            var yp = Predict(X);
            double vloss = loss.Loss(y, yp);
            double vacc = accuracy.Func(y, yp);

            return (vloss, vacc);
        }

        public (double, double) TrainOnBatch(NDarray<double> X, NDarray<double> y)
        {
            var yp = Forward(X);
            double vloss = loss.Loss(y, yp);
            double vacc = accuracy.Func(y, yp);

            var accumGrad = loss.Grad(y, yp);
            Backward(accumGrad);

            return (vloss, vacc);
        }

        public void Summary()
        {
            Console.WriteLine("Summary");
            Console.WriteLine($"Network: {optimizer.Name} / {loss.Name} / {accuracy.Name}");
            Console.WriteLine();

            string inputLayerShape = $"({layers[0].InputShape.Glue()})";

            int nameSize = layers.Max(l => l.Name.Length + 4);
            int paramsSize = layers.Max(l => l.Params.ToString().Length + 4);
            int outputSize = layers.Max(l => l.OutputShape.Glue().Length + 4);
            nameSize = Math.Max(nameSize, "InputLayer".Length + 4);
            paramsSize = Math.Max(paramsSize, "Parameters".Length + 4);
            outputSize = Math.Max(Math.Max(outputSize, "Output".Length + 4), inputLayerShape.Length + 4);

            string fmtHead = $"| {{0,{-nameSize}}}| {{1,{-paramsSize}}}| {{2,{-outputSize}}}|";
            string fmtRow = $"| {{0,{-nameSize}}}|{{1,{paramsSize}}} |{{2,{outputSize}}} |";
            string head = string.Format(fmtHead, 0, 0, 0);
            string sep = Enumerable.Repeat('=', head.Length).Glue("");

            int tot = 0;
            Console.WriteLine(sep);
            Console.WriteLine(fmtHead, "Layer", "Parameters", "Output");
            Console.WriteLine(sep);
            Console.WriteLine(fmtRow, "InputLayer", 0, inputLayerShape);
            foreach (var layer in layers)
            {
                string shout = $"({layer.OutputShape.Glue()})";
                Console.WriteLine(fmtRow, layer.Name, layer.Params, shout);
                tot += layer.Params;
            }

            Console.WriteLine(sep);
            Console.WriteLine();
            Console.WriteLine($"Total Parameters:{tot}");
            Console.WriteLine();
        }

        public void Fit(NDarray<double> trainX, NDarray<double> trainY, int epochs, int batchSize = 50, int displayEpochs = 1, bool shuffle = true)
        {
            var sw = Stopwatch.StartNew();

            for (int k = 0; k <= epochs; ++k)
            {
                List<double> losses = new List<double>();
                List<double> accs = new List<double>();
                var batch = ND.BatchIterator(trainX, trainY, batchSize, shuffle);
                foreach ((var X, var y) in batch)
                {
                    (double vloss, double vacc) = TrainOnBatch(X, y);
                    losses.Add(vloss);
                    accs.Add(vacc);
                }

                if (k % displayEpochs == 0)
                    Console.WriteLine($"Epoch: {k,4}/{epochs}. loss:{losses.Average():0.000000} acc:{accs.Average():0.0000} Time:{sw.ElapsedMilliseconds,10} ms");
            }
            Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");
        }

        public void Fit(NDarray<double> trainX, NDarray<double> trainY, NDarray<double> testX, NDarray<double> testY, int epochs, int batchSize = 50, int displayEpochs = 1, bool shuffle = true)
        {
            var sw = Stopwatch.StartNew();

            for (int k = 0; k <= epochs; ++k)
            {
                List<double> losses = new List<double>();
                List<double> accs = new List<double>();
                var batch = ND.BatchIterator(trainX, trainY, batchSize, shuffle);
                foreach ((var X, var y) in batch)
                {
                    (double vloss, double vacc) = TrainOnBatch(X, y);
                    losses.Add(vloss);
                    accs.Add(vacc);
                }

                if (k % displayEpochs == 0)
                {
                    (double vloss, double vacc) = TestOnBatch(testX, testY);
                    Console.WriteLine($"Epoch: {k,4}/{epochs}. loss:{losses.Average():0.000000} acc:{accs.Average():0.0000}; Validation. loss:{vloss:0.000000} acc:{vacc:0.0000} Time:{sw.ElapsedMilliseconds,10} ms");
                }
            }
            Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");
        }

        public void Test(NDarray<double> testX, NDarray<double> testY)
        {
            (double vloss, double vacc) = TestOnBatch(testX, testY);
            Console.WriteLine($"Test. loss:{vloss:0.000000} acc:{vacc:0.0000}");
        }

        public void ImportWeights(string filename)
        {
            var lstr = File.ReadAllLines(filename);
            Queue<string> q = new Queue<string>(lstr);
            foreach(var layer in layers)
            {
                if (layer is ActivationLayer layer0)
                    continue;

                var w = q.Dequeue();
                var b = q.Dequeue();
                layer.ImportWeights(w, b);
            }
        }
    }
}
