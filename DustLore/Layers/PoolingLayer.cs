using System;
using System.Linq;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public abstract class PoolingLayer
    {

        protected PoolingLayer((int, int) poolShape, int strides, string padding)
        {
            this.poolShape = poolShape;
            this.strides = strides;
            this.padding = padding;
        }

        protected PoolingLayer((int, int) poolShape, (int, int, int) inputShape, int strides, string padding)
        {
            this.poolShape = poolShape;
            this.strides = strides;
            this.padding = padding;

            SetInputShape(new int[] { inputShape.Item1, inputShape.Item2, inputShape.Item3 });
        }

        readonly (int, int) poolShape;
        readonly int strides;
        readonly string padding;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        protected NDarray<double> lastInput, xCol;

        public abstract NDarray<double> PoolForward(NDarray<double> X);
        public abstract NDarray<double> PoolBackward(NDarray<double> X);

        public NDarray<double> Backward(NDarray<double> accumGrad)
        {
            int batchSize = accumGrad.Shape[0];
            int channels = InputShape[0], height = InputShape[1], width = InputShape[2];

            accumGrad = accumGrad.Transpose(2, 3, 0, 1).Reshape(accumGrad.Count);
            var accumGradCol = PoolBackward(accumGrad);

            accumGrad = Images2Columns.Columns2Images2Dfast(accumGradCol, (batchSize * channels, 1, height, width), poolShape, strides, padding);
            accumGrad.ReshapeInplace(batchSize, channels, height, width);

            return accumGrad;
        }

        public NDarray<double> Forward(NDarray<double> X, bool isTraining)
        {
            lastInput = new NDarray<double>(X);
            int batchSize = X.Shape[0], channels = X.Shape[1], height = X.Shape[2], width = X.Shape[3];
            int outHeight = OutputShape[1], outWidth = OutputShape[2];
            X = X.Reshape(batchSize * channels, 1, height, width);
            xCol = Images2Columns.Images2Columns2Dfast(X, poolShape, strides, padding);

            var output = PoolForward(xCol);
            output.ReshapeInplace(outHeight, outWidth, batchSize, channels);
            output = output.Transpose(2, 3, 0, 1);

            return output;
        }

        public int[] GetOutputShape() => OutputShape;

        public void Initialize(IOptimizer optimizer)
        {

        }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
            int channels = InputShape[0], height = InputShape[1], width = InputShape[2];
            int outHeight = (height - poolShape.Item1) / strides + 1;
            int outWidth = (width - poolShape.Item2) / strides + 1;
            OutputShape = new int[] { channels, outHeight, outWidth };
        }
    }
}
