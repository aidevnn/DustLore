﻿using System;
using System.Linq;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public class DropoutLayer : ILayer
    {
        public DropoutLayer(double p = 0.2)
        {
            this.p = p;
        }

        readonly double p;
        public string Name => "DropoutLayer";

        public int Params => 0;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        double[] mask = { 1.0 };

        public NDarray<double> Backward(NDarray<double> accumGrad)
        {
            var accumGrad0 = new NDarray<double>(accumGrad);
            double coef = IsTraining ? 1.0 - p : 1.0;
            for (int i = 0; i < accumGrad0.Count; ++i)
                accumGrad0.Data[i] *= mask[i % mask.Length] * coef;

            return accumGrad0;
        }

        public NDarray<double> Forward(NDarray<double> X, bool isTraining)
        {
            IsTraining = isTraining;
            mask = new double[X.Count];
            var X0 = new NDarray<double>(X);
            double coef = IsTraining ? 1.0 - p : 1.0;
            for (int i = 0; i < X0.Count; ++i)
            {
                var v = mask[i] = Utils.Random.NextDouble() > p ? 1.0 : 0.0;
                X0.Data[i] *= v / coef;
            }

            return X0;
        }

        public int[] GetOutputShape() => OutputShape;

        public void Initialize(IOptimizer optimizer) { }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
            OutputShape = shape.ToArray();
        }

        public void ImportWeights(string w, string b)
        {

        }
    }
}
