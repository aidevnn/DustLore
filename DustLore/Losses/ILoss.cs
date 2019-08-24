using System;
using NDarrayLib;

namespace DustLore.Losses
{
    public interface ILoss
    {
        string Name { get; }
        double Loss(NDarray<double> y, NDarray<double> p);
        NDarray<double> Grad(NDarray<double> y, NDarray<double> p);
    }
}
