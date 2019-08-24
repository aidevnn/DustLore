using System;
using NDarrayLib;

namespace DustLore.Optimizers
{
    public class SGD : IOptimizer
    {
        public SGD(double lr = 0.01, double momentum = 0.0)
        {
            this.lr = lr;
            this.momentum = momentum;
        }

        readonly double lr, momentum;

        public string Name => "SGD";

        public IOptimizer Clone() => new SGD(lr, momentum);

        NDarray<double> wUpdt;
        public void Update(NDarray<double> w, NDarray<double> g)
        {
            if (wUpdt == null)
                wUpdt = new NDarray<double>(w.Shape);

            for (int i = 0; i < wUpdt.Count; ++i)
                wUpdt.Data[i] = momentum * wUpdt.Data[i] + (1 - momentum) * g.Data[i];

            for (int i = 0; i < w.Count; ++i)
                w.Data[i] = w.Data[i] - lr * wUpdt.Data[i];
        }
    }
}
