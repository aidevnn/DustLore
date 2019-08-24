using System;
using System.Linq;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public class DenseLayer : ILayer
    {
        public DenseLayer(int outNodes)
        {
            OutputShape = new int[] { outNodes };
        }

        public DenseLayer(int outNodes, int inputShape)
        {
            OutputShape = new int[] { outNodes };
            InputShape = new int[] { inputShape };
        }

        public string Name => "DenseLayer";

        public int Params => weight.Count + biases.Count;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        NDarray<double> LastInput, weight, biases, wTmp;
        IOptimizer wOpt, bOpt;

        public NDarray<double> Backward(NDarray<double> accumGrad)
        {
            for (int i = 0; i < wTmp.Count; ++i)
                wTmp.Data[i] = weight.Data[i];

            if (IsTraining)
            {
                var gW = ND.GemmTABC(LastInput, accumGrad);
                var gB = ND.SumAxis(accumGrad, 0, true);

                wOpt.Update(weight, gW);
                bOpt.Update(biases, gB);
            }

            return ND.GemmATBC(accumGrad, wTmp);
        }

        public NDarray<double> Forward(NDarray<double> X, bool isTraining)
        {
            IsTraining = isTraining;
            LastInput = new NDarray<double>(X);
            return ND.GemmABC(X, weight, biases);
        }

        public int[] GetOutputShape() => OutputShape;

        public void Initialize(IOptimizer optimizer)
        {
            wOpt = optimizer.Clone();
            bOpt = optimizer.Clone();

            double lim = 2.0 / Math.Sqrt(InputShape[0]);
            weight = ND.Uniform(-lim, lim, InputShape[0], OutputShape[0]);
            biases = new NDarray<double>(1, OutputShape[0]);
            wTmp = new NDarray<double>(weight.Shape);
        }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
        }
    }
}
