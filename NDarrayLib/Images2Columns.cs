using System;
namespace NDarrayLib
{
    public static class Images2Columns
    {

        public static ((int, int), (int, int)) DeterminePadding((int, int) filterShape, string outputShape = "same")
        {
            if (outputShape == "valid")
                return ((0, 0), (0, 0));

            (int filterHeight, int filterWidth) = filterShape;
            int padH1 = (int)(Math.Floor((filterHeight - 1) / 2.0));
            int padH2 = (int)(Math.Ceiling((filterHeight - 1) / 2.0));
            int padW1 = (int)(Math.Floor((filterWidth - 1) / 2.0));
            int padW2 = (int)(Math.Ceiling((filterWidth - 1) / 2.0));

            return ((padH1, padH2), (padW1, padW2));
        }

        static int[] TileRepeatArange(int arange, int repeat, int tile, int coef)
        {
            int dim = arange * repeat * tile;
            int[] data = new int[dim];

            int x = 0;
            for (int i = 0; i < tile; ++i)
                for (int k = 0; k < arange; ++k)
                    for (int j = 0; j < repeat; ++j)
                        data[x++] = coef * k;

            return data;
        }

        static NDarray<int> NDarrayIndices(int[] dx, int[] dy)
        {
            int dimX = dx.Length;
            int dimY = dy.Length;

            var nd = new NDarray<int>(dimY, dimX);
            int r = 0;
            for (int i = 0; i < dy.Length; ++i)
            {
                var dyi = dy[i];
                for (int j = 0; j < dx.Length; ++j)
                    nd.Data[r++] = dyi + dx[j];
            }

            return nd;
        }

        static NDarray<int> TRANDarray(int fh, int fw, int ch, int oh, int ow, int st)
        {
            int[] dx = TileRepeatArange(oh, ow, 1, st);
            int[] dy = TileRepeatArange(fh, fw, ch, 1);
            return NDarrayIndices(dx, dy);
        }

        static NDarray<int> TANDarray(int fh, int fw, int ch, int oh, int ow, int st)
        {
            int[] dx = TileRepeatArange(ow, 1, oh, st);
            int[] dy = TileRepeatArange(fw, 1, fh * ch, 1);
            return NDarrayIndices(dx, dy);
        }

        public static (NDarray<int>, NDarray<int>, NDarray<int>) Im2ColIndices2Dfast(
            (int, int, int, int) imagesShape,
            (int, int) filterShape,
            ((int, int), (int, int)) padding, int stride = 1)
        {
            (int batchSize, int channels, int height, int width) = imagesShape;
            (int filterHeight, int filterWidth) = filterShape;
            ((int, int) padH, (int, int) padW) = padding;
            int outHeight = (int)((height + (padH.Item1 + padH.Item2) - filterHeight) / stride + 1);
            int outWidth = (int)((width + (padW.Item1 + padW.Item2) - filterWidth) / stride + 1);

            var i = TRANDarray(filterHeight, filterWidth, channels, outHeight, outWidth, stride);
            var j = TANDarray(filterHeight, filterWidth, channels, outHeight, outWidth, stride);
            var k = TRANDarray(channels, filterHeight * filterWidth, 1, 1, 1, 1);

            return (k, i, j);
        }

        public static NDarray<double> ImagesPadded2Columns(NDarray<double> nDarray, NDarray<int> ndk, NDarray<int> ndi, NDarray<int> ndj)
        {
            int batchSize = nDarray.Shape[0];
            int channels = nDarray.Shape[1];
            int height = nDarray.Shape[2];
            int width = nDarray.Shape[3];

            int outHeight = ndi.Shape[0];
            int outWidth = ndi.Shape[1];
            int last = batchSize * outWidth;

            var nshape = new int[] { outHeight, last };
            var nd0 = new NDarray<double>(shape: nshape);

            int s00 = channels * height * width, s01 = height * width, s02 = width, s03 = 1;

            for (int h = 0; h < outHeight; ++h)
            {
                for (int w0 = 0; w0 < last; ++w0)
                {
                    int idx0 = h * last + w0;

                    int b = idx0 % batchSize;
                    int w1 = (idx0 / batchSize) % outWidth;

                    var idxK = ndk.Data[h];
                    var idxI = ndi.Data[h * outWidth + w1];
                    var idxJ = ndj.Data[h * outWidth + w1];
                    var idx1 = b * s00 + idxK * s01 + idxI * s02 + idxJ * s03;

                    nd0.Data[idx0] = nDarray.Data[idx1];
                }
            }

            return nd0;
        }

        public static NDarray<double> Images2Columns2Dfast(NDarray<double> images, (int, int) filterShape, int stride = 1, string outputShape = "same")
        {
            (int filterHeight, int filterWidth) = filterShape;
            (var padH, var padW) = DeterminePadding(filterShape, outputShape);
            var imagesPadded = images.Pad((0, 0), (0, 0), padH, padW);

            var sh = images.Shape;
            (var k, var i, var j) = Im2ColIndices2Dfast((sh[0], sh[1], sh[2], sh[3]), filterShape, (padH, padW), stride);

            return ImagesPadded2Columns(imagesPadded, k, i, j);
        }

        public static NDarray<double> Columns2Images2Dfast(NDarray<double> cols, (int, int, int, int) imagesShape, (int, int) filterShape, int stride = 1, string output_shape = "same")
        {
            (int batchSize, int channels, int height, int width) = imagesShape;
            (var padH, var padW) = DeterminePadding(filterShape, output_shape);
            (var ndk, var ndi, var ndj) = Im2ColIndices2Dfast(imagesShape, filterShape, (padH, padW), stride);

            var images = new NDarray<double>(batchSize, channels, height, width);

            int outHeight = ndi.Shape[0];
            int outWidth = ndi.Shape[1];
            int width1 = cols.Count / (outHeight * batchSize);

            int s00 = channels * height * width, s01 = height * width, s02 = width, s03 = 1;
            int s10 = outHeight * width1, s11 = width1, s12 = 1;
            int s20 = width1 * batchSize, s21 = batchSize, s22 = 1;

            for (int b = 0; b < batchSize; ++b)
            {
                for (int h0 = 0; h0 < outHeight; ++h0)
                {
                    for (int w0 = 0; w0 < width1; ++w0)
                    {
                        int idx0 = b * s10 + h0 * s11 + w0 * s12;
                        int idx1 = h0 * s20 + w0 * s21 + b * s22;

                        int h1 = (idx0 / outWidth) % outHeight;
                        int w1 = idx0 % outWidth;

                        var idxK = ndk.Data[h1];
                        var idxI = ndi.Data[h1 * outWidth + w1];
                        var idxJ = ndj.Data[h1 * outWidth + w1];

                        if (idxI >= padH.Item1 && idxI < padH.Item1 + height && idxJ >= padW.Item1 && idxJ < padW.Item1 + width)
                        {
                            int idx2 = b * s00 + idxK * s01 + (idxI - padH.Item1) * s02 + (idxJ - padW.Item1) * s03;
                            images.Data[idx2] = images.Data[idx2] + cols.Data[idx1];
                        }
                    }
                }
            }

            return images;
        }
    }
}
