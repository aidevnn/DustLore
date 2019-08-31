using System;
using System.IO;
using System.Linq;
using NDarrayLib;

namespace DustLore
{
    public static class ImportDataset
    {

        public static (NDarray<double>, NDarray<double>, NDarray<double>, NDarray<double>) DigitsDataset(double ratio, bool normalize = false)
        {
            var raw = File.ReadAllLines("datasets/digits.csv").ToArray();
            var data = raw.SelectMany(l => l.Split(',')).Select(double.Parse).ToArray();
            int dim0 = data.Length / 65;

            var array = Enumerable.Range(0, dim0).Select(i => Enumerable.Range(0, 65).Select(j => data[i * 65 + j]).ToArray()).ToArray();
            var nd0 = new NDarray<double>(array);
            var idx0 = (int)(dim0 * ratio);
            (var X, var y) = nd0.Split(axis: 1, idx: 64);

            double coef = normalize ? 16.0 : 1.0;
            X.ApplyFuncInplace(x => x / coef);
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

        public static (NDarray<double>, NDarray<double>) GenMultiplesSeries(int nums)
        {
            var X = new NDarray<double>(nums, 10, 61);
            var y = new NDarray<double>(nums, 10, 61);

            for (int i = 0; i < nums; ++i)
            {
                int start = Utils.Random.Next(2, 7);
                for (int k = 0; k < 10; ++k)
                {
                    int j0 = (k + 1) * start;
                    int j1 = (k + 2) * start;
                    int idx0 = i * 610 + k * 61 + j0;
                    int idx1 = i * 610 + k * 61 + j1;
                    X.Data[idx0] = 1.0;
                    if (k < 9)
                        y.Data[idx1] = 1.0;
                }

                y.Data[i * 610 + 550] = 1.0;
            }

            return (X, y);
        }

        public static (NDarray<double>, NDarray<double>, NDarray<double>, NDarray<double>) SequenceDataset(int nums, double ratio)
        {
            (var X, var y) = GenMultiplesSeries(nums);
            var idx0 = (int)(nums * ratio);

            (var trainX, var testX) = X.Split(0, idx0);
            (var trainY, var testY) = y.Split(0, idx0);

            Console.WriteLine($"Train on {trainX.Shape[0]} / Test on {testX.Shape[0]}");
            return (trainX, trainY, testX, testY);
        }

    }
}
