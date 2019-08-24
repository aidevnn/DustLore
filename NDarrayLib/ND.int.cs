using System;
namespace NDarrayLib
{
    public static partial class ND
    {
        public static NDarray<int> SumAxis(NDarray<int> nDarray, int axis, bool keepdims = false)
        {
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            var nd = new NDarray<int>(shape: nshape);

            int m = Utils.ArrMul(nDarray.Shape, axis);
            int n = m / nDarray.Shape[axis];
            for (int idx0 = 0; idx0 < nDarray.Count; ++idx0)
            {
                var v = nDarray.Data[idx0];
                int idx1 = (idx0 / m) * n + idx0 % n;
                nd.Data[idx1] += v;
            }

            return nd;
        }

        public static NDarray<int> ProdAxis(NDarray<int> nDarray, int axis, bool keepdims = false)
        {
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            var nd = new NDarray<int>(v0: 1, shape: nshape);

            int m = Utils.ArrMul(nDarray.Shape, axis);
            int n = m / nDarray.Shape[axis];
            for (int idx0 = 0; idx0 < nDarray.Count; ++idx0)
            {
                var v = nDarray.Data[idx0];
                int idx1 = (idx0 / m) * n + idx0 % n;
                nd.Data[idx1] *= v;
            }

            return nd;
        }

        public static NDarray<int> MinAxis(NDarray<int> nDarray, int axis, bool keepdims = false)
        {
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            var nd = new NDarray<int>(v0: int.MaxValue, shape: nshape);

            int m = Utils.ArrMul(nDarray.Shape, axis);
            int n = m / nDarray.Shape[axis];
            for (int idx0 = 0; idx0 < nDarray.Count; ++idx0)
            {
                var v = nDarray.Data[idx0];
                int idx1 = (idx0 / m) * n + idx0 % n;
                nd.Data[idx1] = Math.Min(nd.Data[idx1], v);
            }

            return nd;
        }

        public static NDarray<int> MaxAxis(NDarray<int> nDarray, int axis, bool keepdims = false)
        {
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            var nd = new NDarray<int>(v0: int.MinValue, shape: nshape);

            int m = Utils.ArrMul(nDarray.Shape, axis);
            int n = m / nDarray.Shape[axis];
            for (int idx0 = 0; idx0 < nDarray.Count; ++idx0)
            {
                var v = nDarray.Data[idx0];
                int idx1 = (idx0 / m) * n + idx0 % n;
                nd.Data[idx1] = Math.Max(nd.Data[idx1], v);
            }

            return nd;
        }

        public static NDarray<double> MeanAxis(NDarray<int> nDarray, int axis, bool keepdims = false)
        {
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            var nd = new NDarray<double>(shape: nshape);

            int m = Utils.ArrMul(nDarray.Shape, axis);
            int n = m / nDarray.Shape[axis];
            double coef = nDarray.Shape[axis];
            for (int idx0 = 0; idx0 < nDarray.Count; ++idx0)
            {
                var v = nDarray.Data[idx0];
                int idx1 = (idx0 / m) * n + idx0 % n;
                nd.Data[idx1] += v / coef;
            }

            return nd;
        }

    }
}
