using System;
using System.Linq;
using NDarrayLib;

namespace DustLore.Losses
{
    public class CrossEntropyLoss : ILoss
    {
        public string Name => "CrossEntropyLoss";

        double func(double y, double p)
        {
            var p0 = Math.Min(1 - 1e-15, Math.Max(1e-15, p));
            return -y * Math.Log(p0) - (1 - y) * Math.Log(1 - p0);
        }

        double grad(double y, double p)
        {
            var p0 = Math.Min(1 - 1e-15, Math.Max(1e-15, p));
            return -y / p0 + (1 - y) / (1 - p0);
        }

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
