using System;
using System.Collections.Generic;
using System.Linq;

namespace NDarrayLib
{
    public static class Utils
    {
        public static string Glue<T>(this IEnumerable<T> ts, string sep = " ", string fmt = "{0}") => string.Join(sep, ts.Select(t => string.Format(fmt, t)));

        public static Random Random = new Random();

        public static int ArrMul(int[] shape, int start = 0)
        {
            int a = 1;
            for (int i = start; i < shape.Length; ++i)
                a *= shape[i];

            return a;
        }

        public static int[] Shape2Strides(int[] shape)
        {
            int[] strides = new int[shape.Length];
            int p = 1;
            for (int k = shape.Length - 1; k >= 0; --k)
            {
                strides[k] = p;
                p *= shape[k];
            }

            return strides;
        }

        public static void Int2ArrayIndex(int idx, int[] shape, int[] indices)
        {
            for (int k = shape.Length - 1; k >= 0; --k)
            {
                var sk = shape[k];
                indices[k] = idx % sk;
                idx = idx / sk;
            }
        }

        public static int Int2IntIndex(int idx0, int[] shape, int[] strides)
        {
            int idx1 = 0;
            for (int k = shape.Length - 1; k >= 0; --k)
            {
                var sk = shape[k];
                idx1 += strides[k] * (idx0 % sk);
                idx0 = idx0 / sk;
            }

            return idx1;
        }

        public static int Array2IntIndex(int[] args, int[] shape, int[] strides)
        {
            int idx = 0;
            for (int k = 0; k < args.Length; ++k)
            {
                var v = args[k];
                idx += v * strides[k];
            }

            return idx;
        }

        public static int[] PrepareReshape(int[] baseShape, int[] shape) => PrepareReshape(ArrMul(baseShape), shape);

        public static int[] PrepareReshape(int dim0, int[] shape)
        {
            int mone = shape.Count(i => i == -1);
            if (mone > 1)
                throw new ArgumentException("Can only specify one unknown dimension");

            if (mone == 1)
            {
                int idx = shape.ToList().FindIndex(i => i == -1);
                shape[idx] = 1;
                var dim2 = ArrMul(shape);
                shape[idx] = dim0 / dim2;
            }

            var dim1 = ArrMul(shape);

            if (dim0 != dim1)
                throw new ArgumentException($"cannot reshape array of size {dim0} into shape ({shape.Glue()})");

            return shape;
        }

        public static int[] PrepareTranspose(int rank)
        {
            int[] table = new int[rank];
            for (int i = 0; i < rank; ++i)
                table[i] = rank - i - 1;
            return table;
        }

        public static int[] DoTranspose(int[] arr, int[] table)
        {
            int[] r = new int[arr.Length];
            for (int i = 0; i < arr.Length; ++i)
                r[i] = arr[table[i]];

            return r;
        }

        public static (int, int, int[]) BroadCastShapes(int[] shape0, int[] shape1)
        {
            int sLength0 = shape0.Length;
            int sLength1 = shape1.Length;
            int mLength = Math.Max(sLength0, sLength1);

            int[] nshape = new int[mLength];
            int prv0 = shape0.Last(), prv1 = shape1.Last();
            int nb0 = 0, nb1 = 0;
            for (int k = mLength - 1, i = sLength0 - 1, j = sLength1 - 1; k >= 0; --k, --i, --j)
            {
                int idx0 = i < 0 ? 1 : shape0[i];
                int idx1 = j < 0 ? 1 : shape1[j];
                if (idx0 != idx1 && idx0 != 1 && idx1 != 1)
                    throw new ArgumentException($"Cannot broadcast ({shape0.Glue()}) with ({shape1.Glue()})");

                if ((prv0 == 1 && idx0 != 1) || (prv0 != 1 && idx0 == 1)) ++nb0;
                if ((prv1 == 1 && idx1 != 1) || (prv1 != 1 && idx1 == 1)) ++nb1;
                if ((idx0 == 1 && idx1 == 1) || (prv0 == 1 && prv1 == 1)) { --nb0; --nb1; }

                prv0 = idx0;
                prv1 = idx1;
                nshape[k] = Math.Max(idx0, idx1);
            }

            bool isLeftOne0 = sLength0 < mLength || shape0.First() == 1;
            bool isLeftOne1 = sLength1 < mLength || shape1.First() == 1;
            bool isRightOne0 = sLength1 != 1 && shape0.Last() == 1;
            bool isRightOne1 = sLength1 != 1 && shape1.Last() == 1;
            int sz0 = ArrMul(shape0);
            int sz1 = ArrMul(shape1);

            if (isLeftOne0 && isRightOne0 && sz0 != 1 || nb0 > 1) 
                throw new ArgumentException($"One must be aligned only at one border of shape0 ({shape0.Glue()})");
            if (isLeftOne1 && isRightOne1 && sz1 != 1 || nb1 > 1) 
                throw new ArgumentException($"One must be aligned only at one border of shape1 ({shape1.Glue()})");

            int info0 = isLeftOne0 ? -1 : isRightOne0 ? 1 : 0;
            int info1 = isLeftOne1 ? -1 : isRightOne1 ? 1 : 0;
            return (info0, info1, nshape);
        }

        public static int[] PrepareAxisOps(int[] shape, int axis, bool keepdims)
        {
            int[] nshape = !keepdims ? new int[shape.Length - 1] : shape.ToArray();
            if (!keepdims)
            {
                int j = 0;
                for (int i = 0; i < shape.Length; ++i)
                {
                    if (i == axis) continue;
                    nshape[j++] = shape[i];
                }
            }
            else
                nshape[axis] = 1;

            return nshape;
        }

        public static (int[], int[]) PrepareSplit(int[] shape, int axis, int idx)
        {
            if (axis < 0 || axis >= shape.Length)
                throw new ArgumentException("Bad Split axis");

            int dim = shape[axis];
            if (idx < 0 || idx >= dim)
                throw new ArgumentException("Bad Split index");

            int[] shape0 = shape.ToArray();
            int[] shape1 = shape.ToArray();
            shape0[axis] = idx;
            shape1[axis] -= idx;

            return (shape0, shape1);
        }

        public static (int[], (int, int)[]) PreparePad(int[] shape, (int, int)[] pads)
        {
            if (pads.Length == 1)
                pads = Enumerable.Repeat(pads[0], shape.Length).ToArray();

            if (pads.Length != shape.Length)
                throw new ArgumentException($"Shape and pads must have the same length");

            var nshape = new int[shape.Length];
            for (int i = 0; i < shape.Length; ++i)
            {
                (int padL, int padR) = pads[i];
                nshape[i] = shape[i] + padL + padR;
            }
            return (nshape, pads);
        }

    }
}
