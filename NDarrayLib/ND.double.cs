using System;
namespace NDarrayLib
{
    public static partial class ND
    {
        public static NDarray<double> SumAxis(NDarray<double> nDarray, int axis, bool keepdims = false)
        {
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            var nd = new NDarray<double>(nshape);

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

        public static NDarray<double> ProdAxis(NDarray<double> nDarray, int axis, bool keepdims = false)
        {
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            var nd = new NDarray<double>(v0: 1, shape: nshape);

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

        public static NDarray<double> MinAxis(NDarray<double> nDarray, int axis, bool keepdims = false)
        {
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            var nd = new NDarray<double>(v0: int.MaxValue, shape: nshape);

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

        public static NDarray<double> MaxAxis(NDarray<double> nDarray, int axis, bool keepdims = false)
        {
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            var nd = new NDarray<double>(v0: int.MinValue, shape: nshape);

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

        public static NDarray<double> MeanAxis(NDarray<double> nDarray, int axis, bool keepdims = false)
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

        public static NDarray<double> ApplyFuncAB(NDarray<double> a, NDarray<double> b, Func<double, double, double> func)
        {
            if (a.Count != b.Count)
                throw new Exception();

            NDarray<double> c = new NDarray<double>(a.Shape);
            for (int i = 0; i < a.Count; ++i)
                c.Data[i] = func(a.Data[i], b.Data[i]);

            return c;
        }

        public static void AddAfBinplace(NDarray<double> a, NDarray<double> b, Func<double, double> func)
        {
            if (a.Count != b.Count)
                throw new Exception();

            for (int i = 0; i < a.Count; ++i)
                a.Data[i] += func(b.Data[i]);
        }

        public static void MulAfBinplace(NDarray<double> a, NDarray<double> b, Func<double, double> func)
        {
            if (a.Count != b.Count)
                throw new Exception();

            for (int i = 0; i < a.Count; ++i)
                a.Data[i] += func(b.Data[i]);
        }

        public static NDarray<double> GemmABC(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (ha != wb || (c != null && (c.Shape[0] != 1 || c.Shape[1] != hb)))
                throw new Exception();

            var nd = new NDarray<double>(wa, hb);
            for (int i = 0; i < wa; ++i)
            {
                for (int j = 0; j < hb; ++j)
                {
                    double sum = c != null ? c.Data[j] : 0;
                    for (int k = 0; k < ha; ++k)
                        sum += a.Data[i * ha + k] * b.Data[k * hb + j];

                    nd.Data[i * hb + j] = sum;
                }
            }

            return nd;
        }

        public static NDarray<double> GemmATBC(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (ha != hb || (c != null && (c.Shape[0] != 1 || c.Shape[1] != wb)))
                throw new Exception();

            var nd = new NDarray<double>(wa, wb);
            for (int i = 0; i < wa; ++i)
            {
                for (int j = 0; j < wb; ++j)
                {
                    double sum = c != null ? c.Data[j] : 0;
                    for (int k = 0; k < ha; ++k)
                        sum += a.Data[i * ha + k] * b.Data[j * ha + k];

                    nd.Data[i * wb + j] = sum;
                }
            }

            return nd;
        }

        public static NDarray<double> GemmTABC(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (wa != wb || (c != null && (c.Shape[0] != 1 || c.Shape[1] != hb)))
                throw new Exception();

            var nd = new NDarray<double>(ha, hb);
            for (int i = 0; i < ha; ++i)
            {
                for (int j = 0; j < hb; ++j)
                {
                    double sum = c != null ? c.Data[j] : 0;
                    for (int k = 0; k < wa; ++k)
                        sum += a.Data[k * ha + i] * b.Data[k * hb + j];

                    nd.Data[i * hb + j] = sum;
                }
            }

            return nd;
        }

        public static NDarray<double> GemmTATBC(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (wa != hb || (c != null && (c.Shape[0] != 1 || c.Shape[1] != wb)))
                throw new Exception();

            var nd = new NDarray<double>(ha, wb);
            for (int i = 0; i < ha; ++i)
            {
                for (int j = 0; j < wb; ++j)
                {
                    double sum = c != null ? c.Data[j] : 0;
                    for (int k = 0; k < wa; ++k)
                        sum += a.Data[k * ha + i] * b.Data[j * wa + k];

                    nd.Data[i * wb + j] = sum;
                }
            }

            return nd;
        }

    }
}
