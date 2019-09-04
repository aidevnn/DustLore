using System;
using System.Linq;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public abstract class PoolingLayer
    {

        protected PoolingLayer((int, int) poolShape)
        {
            this.poolShape = poolShape;
        }

        protected readonly (int, int) poolShape;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        protected int[] lastShape;

        public abstract NDarray<double> PoolForward(NDarray<double> X);
        public abstract NDarray<double> PoolBackward(NDarray<double> X);

        public NDarray<double> Backward(NDarray<double> accumGrad)
        {
            return PoolBackward(accumGrad);
        }

        public NDarray<double> Forward(NDarray<double> X, bool isTraining)
        {
            lastShape = X.Shape.ToArray();
            return PoolForward(X);
        }

        public int[] GetOutputShape() => OutputShape;

        public void Initialize(IOptimizer optimizer)
        {

        }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
            int channels = InputShape[0], height = InputShape[1], width = InputShape[2];
            int outHeight = height / poolShape.Item1;
            int outWidth = width / poolShape.Item2;
            OutputShape = new int[] { channels, outHeight, outWidth };
        }

        public void ImportWeights(string w, string b)
        {

        }
    }
}
