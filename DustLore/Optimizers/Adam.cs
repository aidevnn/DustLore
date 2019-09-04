using System;
using NDarrayLib;

namespace DustLore.Optimizers
{
    public class Adam : IOptimizer
    {
        public Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999)
        {
            this.lr = lr;
            this.b1 = b1;
            this.b2 = b2;
        }

        readonly double lr, b1, b2;

        public string Name => "Adam";

        public IOptimizer Clone() => new Adam(lr, b1, b2);

        NDarray<double> m, v;
        public void Update(NDarray<double> w, NDarray<double> g)
        {
            if (m == null)
            {
                m = new NDarray<double>(g.Shape);
                v = new NDarray<double>(g.Shape);
            }

            for (int i = 0; i < m.Count; ++i)
            {
                var g0 = g.Data[i];
                m.Data[i] = b1 * m.Data[i] + (1 - b1) * g0;
                v.Data[i] = b2 * v.Data[i] + (1 - b2) * g0 * g0;
            }

            for (int i = 0; i < m.Count; ++i)
            {
                var mh = m.Data[i] / (1 - b1);
                var vh = v.Data[i] / (1 - b2);
                var w0 = lr * mh / (Math.Sqrt(vh) + 1e-7);

                w.Data[i] -= w0;
            }
        }
    }
}
