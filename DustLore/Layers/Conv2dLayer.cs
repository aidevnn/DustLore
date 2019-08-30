using System;
using System.Linq;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public class Conv2dLayer : ILayer
    {
        public Conv2dLayer(int nfilters, (int, int) filterShape, string padding = "same", int strides = 1)
        {
            this.nfilters = nfilters;
            this.filterShape = filterShape;
            this.padding = padding;
            this.strides = strides;
        }

        public Conv2dLayer(int nfilters, (int, int) filterShape, (int, int, int) inputShape, string padding = "same", int strides = 1)
        {
            this.nfilters = nfilters;
            this.filterShape = filterShape;
            this.inputShape = inputShape;
            this.padding = padding;
            this.strides = strides;

            InputShape = new int[] { inputShape.Item1, inputShape.Item2, inputShape.Item3 };
        }

        readonly int nfilters, strides;
        readonly (int, int) filterShape;
        readonly string padding;

        (int, int, int) inputShape;
        (int, int, int, int) lastInputShape;

        public string Name => "Conv2dLayer";

        public int Params => weight.Count + biases.Count;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        NDarray<double> weight, biases, wCol, xCol;
        IOptimizer wOpt, bOpt;

        public NDarray<double> Backward(NDarray<double> accumGrad)
        {
            accumGrad = accumGrad.Transpose(1, 2, 3, 0).ReshapeInplace(nfilters, -1);

            if (IsTraining)
            {
                var gW = ND.GemmATBC(accumGrad, xCol).ReshapeInplace(weight.Shape);
                var gB = ND.SumAxis(accumGrad, 1, true);

                wOpt.Update(weight, gW);
                bOpt.Update(biases, gB);
            }

            accumGrad = ND.GemmTABC(wCol, accumGrad);
            accumGrad = Images2Columns.Columns2Images2Dfast(accumGrad, lastInputShape, filterShape, strides, padding);
            return accumGrad;
        }

        public NDarray<double> Forward(NDarray<double> X, bool isTraining)
        {
            int batchSize = X.Shape[0], channels = X.Shape[1], height = X.Shape[2], width = X.Shape[3];
            lastInputShape = (batchSize, channels, height, width);
            xCol = Images2Columns.Images2Columns2Dfast(X, filterShape, strides, padding);

            for (int i = 0; i < wCol.Count; ++i)
                wCol.Data[i] = weight.Data[i];

            var output = ND.GemmABC(wCol, xCol);
            output = ND.AddNDarray(output, biases);
            output.ReshapeInplace(OutputShape[0], OutputShape[1], OutputShape[2], batchSize);
            return output.Transpose(3, 0, 1, 2);
        }

        public int[] GetOutputShape()
        {
            (int channels, int height, int width) = inputShape;
            (var padH, var padW) = Images2Columns.DeterminePadding(filterShape, padding);
            int outHeight = (height + padH.Item1 + padH.Item2 - filterShape.Item1) / strides + 1;
            int outWidth = (width + padW.Item1 + padW.Item2 - filterShape.Item2) / strides + 1;
            OutputShape = new int[] { nfilters, outHeight, outWidth };
            return OutputShape;
        }

        public void Initialize(IOptimizer optimizer)
        {
            wOpt = optimizer.Clone();
            bOpt = optimizer.Clone();

            (int filterHeight, int filterWidth) = filterShape;
            int channel = inputShape.Item1;
            double lim = 2.0 / Math.Sqrt(filterHeight * filterWidth);
            weight = ND.Uniform(-lim, lim, nfilters, channel, filterHeight, filterWidth);
            biases = new NDarray<double>(nfilters, 1);
            wCol = new NDarray<double>(nfilters, channel * filterHeight * filterWidth);
        }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
            inputShape = (shape[0], shape[1], shape[2]);
        }
    }
}
