using System;
using NDarrayLib;

namespace DustLore.Layers
{
    public class MaxPooling2dLayer : PoolingLayer, ILayer
    {
        public MaxPooling2dLayer((int, int) poolShape) : base(poolShape) { }

        public string Name => "MaxPooling2dLayer";

        public int Params => 0;

        NDarray<double> argmax;

        public override NDarray<double> PoolBackward(NDarray<double> X)
        {
            int batchSize = lastShape[0], channels = lastShape[1], height = lastShape[2], width = lastShape[3];
            int outHeight = OutputShape[1], outWidth = OutputShape[2];
            var nd = new NDarray<double>(batchSize, channels, height, width);
            int s00 = channels * height * width, s01 = height * width, s02 = width;
            int s10 = channels * outHeight * outWidth, s11 = outHeight * outWidth, s12 = outWidth;

            int remH = height % poolShape.Item1;
            int remW = width % poolShape.Item2;

            for (int b = 0; b < batchSize; ++b)
            {
                for (int c = 0; c < channels; ++c)
                {
                    for (int h = 0; h < height; ++h)
                    {
                        for (int w = 0; w < width; ++w)
                        {
                            int idx0 = b * s00 + c * s01 + h * s02 + w;
                            int h0 = (h + remH) / poolShape.Item1;
                            int w0 = (w + remW) / poolShape.Item2;
                            int idx1 = b * s10 + c * s11 + h0 * s12 + w0;
                            nd.Data[idx0] = X.Data[idx1];
                        }
                    }
                }
            }

            //nd = ND.MulNDarray(nd, argmax);
            return nd;
        }

        public override NDarray<double> PoolForward(NDarray<double> X)
        {
            int batchSize = X.Shape[0], channels = X.Shape[1], height = X.Shape[2], width = X.Shape[3];
            int outHeight = OutputShape[1], outWidth = OutputShape[2];
            var nd = new NDarray<double>(batchSize, channels, outHeight, outWidth);
            argmax = new NDarray<double>(batchSize, channels, height, width);
            int s00 = channels * height * width, s01 = height * width, s02 = width;
            int s10 = channels * outHeight * outWidth, s11 = outHeight * outWidth, s12 = outWidth;

            int remH = height % poolShape.Item1;
            int remW = width % poolShape.Item2;

            for (int b = 0; b < batchSize; ++b)
            {
                for (int c = 0; c < channels; ++c)
                {
                    for (int h = 0; h < outHeight; ++h)
                    {
                        for (int w = 0; w < outWidth; ++w)
                        {
                            int idx0 = b * s10 + c * s11 + h * s12 + w;
                            double max = double.MinValue;
                            int idxMax = 0;
                            int offsetH = h == 0 ? 0 : remH;
                            int offsetW = w == 0 ? 0 : remW;
                            for (int i = 0; i < poolShape.Item1; ++i)
                            {
                                for (int j = 0; j < poolShape.Item2; ++j)
                                {
                                    int idx1 = b * s00 + c * s01 + (h * poolShape.Item1 + i + offsetH) * s02 + (w * poolShape.Item2 + j + offsetW);
                                    var v = X.Data[idx1];
                                    if (v > max)
                                    {
                                        max = v;
                                        idxMax = idx1;
                                    }
                                }
                            }

                            for (int i = 0; i < poolShape.Item1; ++i)
                            {
                                for (int j = 0; j < poolShape.Item2; ++j)
                                {
                                    int idx1 = b * s00 + c * s01 + (h * poolShape.Item1 + i + offsetH) * s02 + (w * poolShape.Item2 + j + offsetW);
                                    if (Math.Abs(max - X.Data[idx1]) < 1e-6)
                                        argmax.Data[idx1] = 1.0;
                                }
                            }

                            //argmax.Data[idxMax] = 1.0;
                            nd.Data[idx0] = max;
                        }
                    }
                }
            }

            return nd;
        }
    }
}
