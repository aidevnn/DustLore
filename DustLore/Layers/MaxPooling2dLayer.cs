using System;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public class MaxPooling2dLayer : PoolingLayer, ILayer
    {
        public MaxPooling2dLayer((int, int) poolShape, int strides, string padding) 
            : base(poolShape, strides, padding) { }

        public MaxPooling2dLayer((int, int) poolShape, (int, int, int) inputShape, int strides, string padding) 
            : base(poolShape, inputShape, strides, padding) { }


        public string Name => "MaxPooling2dLayer";

        NDarray<double> cache;

        public int Params => throw new NotImplementedException();

        public override NDarray<double> PoolBackward(NDarray<double> X)
        {
            throw new NotImplementedException();
        }

        public override NDarray<double> PoolForward(NDarray<double> X)
        {
            throw new NotImplementedException();
        }
    }
}
