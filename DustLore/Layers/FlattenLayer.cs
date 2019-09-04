using System;
using System.Linq;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public class FlattenLayer : ILayer
    {
        public FlattenLayer()
        {
        }

        public FlattenLayer(int[] inputShape)
        {
            InputShape = inputShape.ToArray();
            int count = Utils.ArrMul(InputShape);
            OutputShape = new int[] { count };
        }

        public string Name => "FlattenLayer";

        public int Params => 0;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        public int[] LastShape;

        public NDarray<double> Backward(NDarray<double> accumGrad)
        {
            return accumGrad.Reshape(LastShape);
        }

        public NDarray<double> Forward(NDarray<double> X, bool isTraining)
        {
            LastShape = X.Shape.ToArray();
            return X.Reshape(X.Shape[0], -1);
        }

        public int[] GetOutputShape() => OutputShape;

        public void Initialize(IOptimizer optimizer) { }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
            int count = Utils.ArrMul(InputShape);
            OutputShape = new int[] { count };
        }

        public void ImportWeights(string w, string b)
        {

        }
    }
}
