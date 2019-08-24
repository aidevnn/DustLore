using System;
using NDarrayLib;

namespace DustLore.Activations
{
    public class SigmoidActivation : IActivation
    {
        public string Name => "Sigmoid";

        double func(double x) => 1.0 / (1.0 + Math.Exp(-x));
        double grad(double x)
        {
            var x0 = func(x);
            return x0 * (1.0 - x0);
        }

        public NDarray<double> Func(NDarray<double> X) => X.ApplyFunc(func);
        public NDarray<double> Grad(NDarray<double> X) => X.ApplyFunc(grad);
    }
}
