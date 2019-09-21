using System;
using System.Linq;

namespace NDarrayLib
{
    public enum Backend { CSharp, MKL }
    public static partial class ND
    {
        public static Backend Backend = Backend.CSharp;
        public static NDarray<double> AddNDarray(NDarray<double> a, NDarray<double> b, double ca = 1, double cb = 1)
        {
            int acount = a.Count;
            int bcount = b.Count;

            (int ea, int eb, int[] nshape) = Utils.BroadCastShapes(a.Shape, b.Shape);
            NDarray<double> r = new NDarray<double>(nshape);

            int diva = ea <= 0 ? 1 : r.Count / acount;
            int moda = ea >= 0 ? r.Count : acount;
            int divb = eb <= 0 ? 1 : r.Count / bcount;
            int modb = eb >= 0 ? r.Count : bcount;

            for (int i = 0; i < r.Count; ++i)
            {
                int ia = (i / diva) % moda;
                int ib = (i / divb) % modb;
                r.Data[i] = ca * a.Data[ia] + cb * b.Data[ib];
            }

            return r;
        }

        public static NDarray<double> SubNDarray(NDarray<double> a, NDarray<double> b, double ca = 1, double cb = 1) => AddNDarray(a, b, ca, -cb);

        public static NDarray<double> MulNDarray(NDarray<double> a, NDarray<double> b, double ca = 1, double cb = 1)
        {
            int acount = a.Count;
            int bcount = b.Count;

            (int ea, int eb, int[] nshape) = Utils.BroadCastShapes(a.Shape, b.Shape);
            NDarray<double> r = new NDarray<double>(nshape);

            int diva = ea <= 0 ? 1 : r.Count / acount;
            int moda = ea >= 0 ? r.Count : acount;
            int divb = eb <= 0 ? 1 : r.Count / bcount;
            int modb = eb >= 0 ? r.Count : bcount;

            for (int i = 0; i < r.Count; ++i)
            {
                int ia = (i / diva) % moda;
                int ib = (i / divb) % modb;
                r.Data[i] = ca * a.Data[ia] * cb * b.Data[ib];
            }

            return r;
        }

        public static NDarray<double> DivNDarray(NDarray<double> a, NDarray<double> b, double ca = 1, double cb = 1)
        {
            int acount = a.Count;
            int bcount = b.Count;

            (int ea, int eb, int[] nshape) = Utils.BroadCastShapes(a.Shape, b.Shape);
            NDarray<double> r = new NDarray<double>(nshape);

            int diva = ea <= 0 ? 1 : r.Count / acount;
            int moda = ea >= 0 ? r.Count : acount;
            int divb = eb <= 0 ? 1 : r.Count / bcount;
            int modb = eb >= 0 ? r.Count : bcount;

            for (int i = 0; i < r.Count; ++i)
            {
                int ia = (i / diva) % moda;
                int ib = (i / divb) % modb;
                r.Data[i] = (ca * a.Data[ia]) / (cb * b.Data[ib]);
            }

            return r;
        }

        public static NDarray<double> SumAxis(NDarray<double> nDarray, int axis, bool keepdims = false)
        {
            axis = (axis + nDarray.Shape.Length) % nDarray.Shape.Length;
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
            axis = (axis + nDarray.Shape.Length) % nDarray.Shape.Length;
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
            axis = (axis + nDarray.Shape.Length) % nDarray.Shape.Length;
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
            axis = (axis + nDarray.Shape.Length) % nDarray.Shape.Length;
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
            axis = (axis + nDarray.Shape.Length) % nDarray.Shape.Length;
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

        public static NDarray<double> VarAxis(NDarray<double> nDarray, int axis, bool keepdims = false)
        {
            axis = (axis + nDarray.Shape.Length) % nDarray.Shape.Length;
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            var nd = new NDarray<double>(shape: nshape);

            int m = Utils.ArrMul(nDarray.Shape, axis);
            int n = m / nDarray.Shape[axis];
            double coef = nDarray.Shape[axis];

            double[] sum = new double[nd.Count];
            double[] mean = new double[nd.Count];

            for (int idx0 = 0; idx0 < nDarray.Count; ++idx0)
            {
                int idx1 = (idx0 / m) * n + idx0 % n;
                var x = nDarray.Data[idx0];
                mean[idx1] += x / coef;
                sum[idx1] += x * x / coef;
            }

            for (int idx = 0; idx < nd.Count; ++idx)
            {
                var s0 = sum[idx];
                var m0 = mean[idx];
                nd.Data[idx] = s0 - m0 * m0;
            }

            return nd;
        }

        public static NDarray<int> ArgmaxAxis(NDarray<double> nDarray, int axis, bool keepdims = false)
        {
            axis = (axis + nDarray.Shape.Length) % nDarray.Shape.Length;
            var nshape = Utils.PrepareAxisOps(nDarray.Shape, axis, keepdims);
            var nd = new NDarray<int>(shape: nshape);

            int[] nshape0 = nDarray.Shape.ToArray();
            nshape0[axis] = 1;
            int m = Utils.ArrMul(nshape0, axis);
            int n = nDarray.Shape[axis];

            for (int idx0 = 0; idx0 < nd.Count; ++idx0)
            {
                int start = (idx0 / m) * m * n + (idx0 % m);
                int bIdx = 0;
                double bVal = double.MinValue;
                for (int k = 0; k < n; ++k)
                {
                    int idx1 = start + k * m;
                    var v = nDarray.Data[idx1];
                    if (v > bVal)
                    {
                        bVal = v;
                        bIdx = k;
                    }
                }

                nd.Data[idx0] = bIdx;
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
            if (Backend == Backend.MKL)
                return NDmkl.GemmABC(a, b, c);

            return NDsharp.GemmABC(a, b, c);
        }

        public static NDarray<double> GemmATBC(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (Backend == Backend.MKL)
                return NDmkl.GemmATBC(a, b, c);

            return NDsharp.GemmATBC(a, b, c);
        }

        public static NDarray<double> GemmTABC(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (Backend == Backend.MKL)
                return NDmkl.GemmTABC(a, b, c);

            return NDsharp.GemmTABC(a, b, c);
        }

        public static NDarray<double> GemmTATBC(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (Backend == Backend.MKL)
                return NDmkl.GemmTATBC(a, b, c);

            return NDsharp.GemmTATBC(a, b, c);
        }

    }
}
