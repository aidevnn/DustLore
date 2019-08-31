using System;
using NDarrayLib;

namespace DustLore.Layers
{
    public class MaxPooling2dLayer : PoolingLayer, ILayer
    {
        public MaxPooling2dLayer((int, int) poolShape, int strides = 1, string padding = "valid") 
            : base(poolShape, strides, padding) { }

        public MaxPooling2dLayer((int, int) poolShape, (int, int, int) inputShape, int strides = 1, string padding = "valid") 
            : base(poolShape, inputShape, strides, padding) { }

        public string Name => "MaxPooling2dLayer";

        public int Params => 0;

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
