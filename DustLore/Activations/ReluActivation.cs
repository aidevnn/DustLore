using System;
using System.Linq;
using NDarrayLib;

namespace DustLore.Activations
{
    public class ReluActivation : IActivation
    {
        public string Name => "Relu";

        public NDarray<double> Func(NDarray<double> X)
        {
            var data = X.Data.Select(x => x >= 0.0 ? x : 0.0).ToArray();
            var r = new NDarray<double>(data: data, shape: X.Shape);
            return r;
        }

        public NDarray<double> Grad(NDarray<double> X)
        {
            var data = X.Data.Select(x => x >= 0.0 ? 1.0 : 0.0).ToArray();
            var r = new NDarray<double>(data: data, shape: X.Shape);
            return r;
        }
    }
}
