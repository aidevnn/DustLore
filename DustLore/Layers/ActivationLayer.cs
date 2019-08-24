using System;
using System.Linq;
using DustLore.Activations;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public class ActivationLayer : ILayer
    {
        public ActivationLayer(IActivation activation)
        {
            this.activation = activation;
        }

        readonly IActivation activation;

        public string Name => $"{activation.Name}Layer";

        public int Params => 0;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        NDarray<double> LastInput;

        public NDarray<double> Backward(NDarray<double> accumGrad)
        {
            var x = activation.Grad(LastInput);
            for (int i = 0; i < accumGrad.Count; ++i)
                accumGrad.Data[i] *= x.Data[i];

            return accumGrad;
        }

        public NDarray<double> Forward(NDarray<double> X, bool isTraining)
        {
            LastInput = new NDarray<double>(X);
            return activation.Func(X);
        }

        public int[] GetOutputShape() => OutputShape;

        public void Initialize(IOptimizer optimizer) { }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
            OutputShape = shape.ToArray();
        }
    }

    public class SigmoidLayer : ActivationLayer
    {
        public SigmoidLayer() : base(new SigmoidActivation()) { }
    }

    public class TanhLayer : ActivationLayer
    {
        public TanhLayer() : base(new TanhActivation()) { }
    }
}
