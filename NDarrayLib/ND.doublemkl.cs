using System;
using System.Runtime.InteropServices;

namespace NDarrayLib
{
    public static class NDmkl
    {
        [DllImport("mkl_rt", EntryPoint = "cblas_dgemm")]
        static extern void MKL_dgemm(int Order, int TransA, int TransB,
            int M, int N, int K,
            double alpha, double[] A, int lda, double[] B, int ldb,
            double beta, double[] C, int ldc);

        public static NDarray<double> GemmABC(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (ha != wb || (c != null && (c.Count != wa && c.Count != hb && c.Count != wa * hb)))
                throw new Exception();

            var nd = new NDarray<double>(wa, hb);
            if (c != null)
                nd = ND.AddNDarray(nd, c);

            //MKL_dgemm(101, 111, 111, m', n', k, 1.0, A, k, B, n, 0.0, C, n'); A=m' x k; B = k x n'; C = m' x n'
            MKL_dgemm(101, 111, 111, wa, hb, ha, 1.0, a.Data, ha, b.Data, hb, 1.0, nd.Data, hb);

            return nd;
        }

        public static NDarray<double> GemmATBC(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (ha != hb || (c != null && (c.Count != wa && c.Count != wb && c.Count != wa * wb)))
                throw new Exception();

            var nd = new NDarray<double>(wa, wb);
            if (c != null)
                nd = ND.AddNDarray(nd, c);

            //MKL_dgemm(101, 111, 111, m', n', k, 1.0, A, k, B, n, 0.0, C, n'); A=m' x k; B = k x n'; C = m' x n'
            MKL_dgemm(101, 111, 112, wa, wb, ha, 1.0, a.Data, ha, b.Data, hb, 1.0, nd.Data, wb);

            return nd;
        }

        public static NDarray<double> GemmTABC(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (wa != wb || (c != null && (c.Count != ha && c.Count != hb && c.Count != ha * hb)))
                throw new Exception();

            var nd = new NDarray<double>(ha, hb);
            if (c != null)
                nd = ND.AddNDarray(nd, c);

            //MKL_dgemm(101, 111, 111, m', n', k, 1.0, A, k, B, n, 0.0, C, n'); A=m' x k; B = k x n'; C = m' x n'
            MKL_dgemm(101, 112, 111, ha, hb, wa, 1.0, a.Data, ha, b.Data, hb, 1.0, nd.Data, hb);

            return nd;
        }

        public static NDarray<double> GemmTATBC(NDarray<double> a, NDarray<double> b, NDarray<double> c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (wa != hb || (c != null && (c.Count != ha && c.Count != wb && c.Count != ha * wb)))
                throw new Exception();

            var nd = new NDarray<double>(ha, wb);
            if (c != null)
                nd = ND.AddNDarray(nd, c);

            //MKL_dgemm(101, 111, 111, m', n', k, 1.0, A, k, B, n, 0.0, C, n'); A=m' x k; B = k x n'; C = m' x n'
            MKL_dgemm(101, 112, 112, ha, wb, wa, 1.0, a.Data, ha, b.Data, hb, 1.0, nd.Data, wb);

            return nd;
        }

    }
}
