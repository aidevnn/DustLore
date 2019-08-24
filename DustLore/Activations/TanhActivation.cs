using System;
using NDarrayLib;

namespace DustLore.Activations
{
    public class TanhActivation : IActivation
    {
        public string Name => "Tanh";

        double func(double x) => Math.Tanh(x);
        double grad(double x)
        {
            var x0 = Math.Tanh(x);
            return 1.0 - x0 * x0;
        }

        public NDarray<double> Func(NDarray<double> X) => X.ApplyFunc(func);
        public NDarray<double> Grad(NDarray<double> X) => X.ApplyFunc(grad);
    }
}
