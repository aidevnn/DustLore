using System;
using System.IO;
using System.Linq;
using NDarrayLib;

namespace DustLore
{
    public static class ImportDataset
    {

        public static (NDarray<double>, NDarray<double>, NDarray<double>, NDarray<double>) DigitsDataset(double ratio)
        {
            var raw = File.ReadAllLines("datasets/digits.csv").ToArray();
            var data = raw.SelectMany(l => l.Split(',')).Select(double.Parse).ToArray();
            int dim0 = data.Length / 65;

            var array = Enumerable.Range(0, dim0).Select(i => Enumerable.Range(0, 65).Select(j => data[i * 65 + j]).ToArray()).ToArray();
            var nd0 = new NDarray<double>(array);
            var idx0 = (int)(dim0 * ratio);
            (var X, var y) = nd0.Split(axis: 1, idx: 64);

            X.ApplyFuncInplace(x => x / 16.0);
            var yd = y.Data.Select(i => { var d = new double[10]; d[Convert.ToInt32(i)] = 1; return d; }).ToArray();
            y = new NDarray<double>(yd);

            (var trainX, var testX) = X.Split(axis: 0, idx: idx0);
            (var trainY, var testY) = y.Split(axis: 0, idx: idx0);

            Console.WriteLine($"Train on {trainX.Shape[0]} / Test on {testX.Shape[0]}");
            return (trainX, trainY, testX, testY);
        }

        public static (NDarray<double>, NDarray<double>, NDarray<double>, NDarray<double>) IrisDataset(double ratio)
        {
            var raw = File.ReadAllLines("datasets/iris.csv").ToArray();
            var data = raw.SelectMany(l => l.Split(',')).Select(double.Parse).ToArray();
            int dim0 = data.Length / 7;

            var array = Enumerable.Range(0, dim0).Select(i => Enumerable.Range(0, 7).Select(j => data[i * 7 + j]).ToArray()).ToArray();
            var nd = new NDarray<double>(array);

            var idx0 = (int)(dim0 * ratio);
            (var train, var test) = nd.Split(axis: 0, idx: idx0);

            (var trainX, var trainY) = train.Split(axis: 1, idx: 4);
            (var testX, var testY) = test.Split(axis: 1, idx: 4);

            var vmax = Enumerable.Range(0, 4).Select(i => array.Max(a => a[i])).ToArray();
            trainX.ApplyFuncInplace((i, x) => x / vmax[i % 4]);
            testX.ApplyFuncInplace((i, x) => x / vmax[i % 4]);

            Console.WriteLine($"Train on {trainX.Shape[0]} / Test on {testX.Shape[0]}");
            return (trainX, trainY, testX, testY);
        }
    }
}
