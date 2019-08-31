using System;
using NDarrayLib;

namespace DustLore.Layers
{
    public class AveragePooling2dLayer : PoolingLayer, ILayer
    {
        public AveragePooling2dLayer((int, int) poolShape, int strides = 1, string padding = "valid") 
            : base(poolShape, strides, padding) { }

        public AveragePooling2dLayer((int, int) poolShape, (int, int, int) inputShape, int strides = 1, string padding = "valid")
            : base(poolShape, inputShape, strides, padding) { }

        public string Name => "AveragePooling2dLayer";

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