using System;
using NDarrayLib;
namespace DustLore.Optimizers
{
    public interface IOptimizer
    {
        string Name { get; }
        IOptimizer Clone();
        void Update(NDarray<double> w, NDarray<double> g);
    }
}
