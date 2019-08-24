using System;
using System.Collections.Generic;
using System.Linq;

namespace NDarrayLib
{
    public static partial class ND
    {
        public static NDarray<U> Uniform<U>(U min, U max, params int[] shape)
        {
            NDarray<U> nd = new NDarray<U>(shape);
            for (int i = 0; i < nd.Count; ++i)
                nd.Data[i] = NDarray<U>.OpsT.Rand(min, max);

            return nd;
        }

        public static List<(NDarray<U>, NDarray<U>)> BatchIterator<U>(NDarray<U> X, NDarray<U> y, int batchsize, bool shuffle = true)
        {
            int dim0 = X.Shape[0];
            batchsize = dim0 < batchsize ? dim0 : batchsize;
            int nb = dim0 / batchsize;

            Queue<int> q = new Queue<int>(Enumerable.Range(0, dim0).OrderBy(c => shuffle ? Utils.Random.NextDouble() : 0));
            List<(NDarray<U>, NDarray<U>)> batch = new List<(NDarray<U>, NDarray<U>)>();
            var xshape = X.Shape.ToArray();
            var yshape = y.Shape.ToArray();
            xshape[0] = batchsize;
            yshape[0] = batchsize;

            int xs = Utils.ArrMul(xshape, 1);
            int ys = Utils.ArrMul(yshape, 1);

            for (int k = 0; k < nb; ++k)
            {
                var lx = new NDarray<U>(xshape);
                var ly = new NDarray<U>(yshape);
                for (int i = 0; i < batchsize; ++i)
                {
                    int idx = q.Dequeue();
                    X.GetAtIndex(idx).CopyTo(lx.Data, i * xs);
                    y.GetAtIndex(idx).CopyTo(ly.Data, i * ys);
                }
                batch.Add((lx, ly));
            }

            return batch;
        }

    }
}
