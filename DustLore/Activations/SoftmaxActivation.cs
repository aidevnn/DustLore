using System;
using System.Linq;
using NDarrayLib;

namespace DustLore.Activations
{
    public class SoftmaxActivation : IActivation
    {
        public string Name => "Softmax";

        public NDarray<double> Func(NDarray<double> X)
        {
            int axis = X.Shape.Length - 1;
            int shapeAxis = X.Shape[axis];
            var mx = ND.MaxAxis(X, axis, true);
            double[] data0 = new double[X.Count];
            for (int i = 0; i < X.Count; ++i)
                data0[i] = Math.Exp(X.Data[i] - mx.Data[i / shapeAxis]);

            var ex = new NDarray<double>(data: data0, shape: X.Shape);
            var sx = ND.SumAxis(ex, axis, true);
            for (int i = 0; i < X.Count; ++i)
                ex.Data[i] /= sx.Data[i / shapeAxis];

            return ex;
        }

        public NDarray<double> Grad(NDarray<double> X)
        {
            var p = Func(X);
            p.ApplyFuncInplace(x => x * (1 - x));
            return p;
        }
    }
}
