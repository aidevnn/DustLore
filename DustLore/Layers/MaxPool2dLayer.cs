using System;
using System.Linq;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public class MaxPool2dLayer : ILayer
    {
        public MaxPool2dLayer((int, int) poolShape)
        {
            this.poolShape = poolShape;
            stride = (poolShape.Item1 + poolShape.Item2) / 2;
        }

        public MaxPool2dLayer((int, int) poolShape, int stride)
        {
            this.poolShape = poolShape;
            this.stride = stride;
        }

        readonly (int, int) poolShape;
        readonly int stride;
        int[] argMax;

        public string Name => "MaxPool2dLayer";

        public int Params => 0;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        public NDarray<double> Backward(NDarray<double> accumGrad)
        {
            /*
        batch_size, _, _, _ = accum_grad.shape
        channels, height, width = self.input_shape
        accum_grad = accum_grad.transpose(2, 3, 0, 1).ravel()

        # MaxPool or AveragePool specific method
        accum_grad_col = self._pool_backward(accum_grad)
            def _pool_backward(self, accum_grad):
                accum_grad_col = np.zeros((np.prod(self.pool_shape), accum_grad.size))
                arg_max = self.cache
                accum_grad_col[arg_max, range(accum_grad.size)] = accum_grad
                return accum_grad_col

        accum_grad = column_to_image(accum_grad_col, (batch_size * channels, 1, height, width), self.pool_shape, self.stride, 0)
        accum_grad = accum_grad.reshape((batch_size,) + self.input_shape)

        return accum_grad
             */


            int batch_size = accumGrad.Shape[0];
            int channels = InputShape[0], height = InputShape[1], width = InputShape[2];
            accumGrad = accumGrad.Transpose(2, 3, 0, 1);
            var accum_grad_col = new NDarray<double>(poolShape.Item1 * poolShape.Item2, accumGrad.Count);

            foreach(var i in argMax)
            {
                for(int j = 0; j < accumGrad.Count; ++j)
                {
                    int idx0 = i * accumGrad.Count + j;
                    accum_grad_col.Data[idx0] = accumGrad.Data[j];
                }
            }

            accumGrad = Images2Columns.Columns2Images2Dfast(accum_grad_col, (batch_size * channels, 1, height, width), poolShape, stride, "valid");
            accumGrad = accumGrad.Reshape(batch_size, channels, height, width);
            return accumGrad;
        }

        public NDarray<double> Forward(NDarray<double> X, bool isTraining)
        {
            int batch_size = X.Shape[0], channels = X.Shape[1], height = X.Shape[2], width = X.Shape[3];
            int out_height = OutputShape[1], out_width = OutputShape[2];
            X = X.Reshape(batch_size * channels, 1, height, width);
            var X_col = Images2Columns.Images2Columns2Dfast(X, poolShape, stride, "valid");

            argMax = ND.ArgmaxAxis(X_col, 0).Data;
            var output = ND.MaxAxis(X_col, 0);
            output = output.Reshape(out_height, out_width, batch_size, channels);
            output = output.Transpose(2, 3, 0, 1);

            return output;
        }

        public int[] GetOutputShape() => OutputShape;

        public void ImportWeights(string w, string b)
        {

        }

        public void Initialize(IOptimizer optimizer)
        {

        }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
            int channels = InputShape[0], height = InputShape[1], width = InputShape[2];
            int outHeight = (height - poolShape.Item1) / stride + 1;
            int outWidth = (width - poolShape.Item2) / stride + 1;
            OutputShape = new int[] { channels, outHeight, outWidth };
        }
    }
}
