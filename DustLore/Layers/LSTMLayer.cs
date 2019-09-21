using System;
using System.Collections.Generic;
using System.Linq;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public class LSTMLayer : ILayer
    {
        public LSTMLayer()
        {
        }

        public string Name => "LSTMLayer";

        public int Params => throw new NotImplementedException();

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        public NDarray<double> Backward(NDarray<double> accumGrad)
        {
            throw new NotImplementedException();
        }

        public NDarray<double> Forward(NDarray<double> X, bool isTraining)
        {
            throw new NotImplementedException();
        }

        public int[] GetOutputShape()
        {
            throw new NotImplementedException();
        }

        public void ImportWeights(string w, string b)
        {
            throw new NotImplementedException();
        }

        public void Initialize(IOptimizer optimizer)
        {
            throw new NotImplementedException();
        }

        public void SetInputShape(int[] shape)
        {
            throw new NotImplementedException();
        }
    }
}
