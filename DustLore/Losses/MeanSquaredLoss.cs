using System;
using System.Linq;
using NDarrayLib;

namespace DustLore.Losses
{
    public class MeanSquaredLoss : ILoss
    {
        public string Name => "MeanSquaredLoss";

        double func(double y, double p) => 0.5 * (p - y) * (p - y);

        double grad(double y, double p) => p - y;

        public NDarray<double> Grad(NDarray<double> y, NDarray<double> p) => ND.ApplyFuncAB(y, p, grad);

        public double Loss(NDarray<double> y, NDarray<double> p)
        {
            double[] c = new double[y.Count];
            for (int i = 0; i < c.Length; ++i)
                c[i] = func(y.Data[i], p.Data[i]);

            return c.Average();
        }
    }
}
