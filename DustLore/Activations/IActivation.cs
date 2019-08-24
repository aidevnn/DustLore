using System;
using NDarrayLib;
namespace DustLore.Activations
{
    public interface IActivation
    {
        string Name { get; }
        NDarray<double> Func(NDarray<double> X);
        NDarray<double> Grad(NDarray<double> X);
    }
}
