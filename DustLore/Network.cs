using System;
using System.Collections.Generic;
using System.Diagnostics;
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

        public void Summary(bool shape = false)
        {
            Console.WriteLine("Summary");
            Console.WriteLine($"Network: {optimizer.Name} / {loss.Name} / {accuracy.Name}");
            Console.WriteLine($"Input  Shape:({layers[0].InputShape.Glue()})");
            int tot = 0;
            foreach (var layer in layers)
            {
                string shin = shape ? $"({layer.InputShape.Glue()})" : $"{Utils.ArrMul(layer.InputShape),5}";
                shin = shape ? $"{shin,10}" : shin;
                string shout = shape ? $"({layer.OutputShape.Glue()})" : $"{Utils.ArrMul(layer.OutputShape),5}";
                shout = shape ? $"{shout,10}" : shout;
                Console.WriteLine($"Layer: {layer.Name,-20} Parameters: {layer.Params,7} Nodes[In:{shin} -> Out:{shout}]");
                tot += layer.Params;
            }

            Console.WriteLine($"Output Shape:({layers.Last().OutputShape.Glue()})");
            Console.WriteLine($"Total Parameters:{tot}");
            Console.WriteLine();
        }

        public void Fit(NDarray<double> trainX, NDarray<double> trainY, int epochs, int displayEpochs, int batchSize = 50, bool shuffle = true)
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
    }
}
