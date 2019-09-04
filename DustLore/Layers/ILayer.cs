using System;
using DustLore.Activations;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public interface ILayer
    {
        string Name { get; }
        int Params { get; }
        bool IsTraining { get; set; }
        int[] InputShape { get; set; }
        int[] OutputShape { get; set; }

        void SetInputShape(int[] shape);
        int[] GetOutputShape();

        NDarray<double> Backward(NDarray<double> accumGrad);
        NDarray<double> Forward(NDarray<double> X, bool isTraining);
        void Initialize(IOptimizer optimizer);
        void ImportWeights(string w, string b);
    }
}
