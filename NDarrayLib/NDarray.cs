using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NDarrayLib
{
    public class NDarray<U>
    {
        public static Operations<U> OpsT;

        static NDarray()
        {
            if (typeof(U) == typeof(int))
                OpsT = new OpsInt() as Operations<U>;
            else if (typeof(U) == typeof(float))
                OpsT = new OpsFloat() as Operations<U>;
            else if (typeof(U) == typeof(double))
                OpsT = new OpsDouble() as Operations<U>;
            else
                throw new ArgumentException($"{typeof(U).Name} is not supported. Only int, float or double");
        }

        public int[] Shape { get; protected set; }
        public int Count { get; protected set; }

        public U[] Data;

        public NDarray(params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { 1 };

            Shape = shape.ToArray();
            Count = Utils.ArrMul(Shape);

            Data = new U[Count];
        }

        public NDarray(U v0, params int[] shape)
        {
            Shape = shape.ToArray();
            Count = Utils.ArrMul(Shape);
            Data = Enumerable.Repeat(v0, Count).ToArray();
            if (Data.Length != Count)
                throw new Exception();
        }

        public NDarray(U[] data, params int[] shape)
        {
            Shape = shape.ToArray();
            Count = Utils.ArrMul(Shape);
            Data = data;
            if (Data.Length != Count)
                throw new Exception();
        }

        public NDarray(U[][] data)
        {
            Data = data.SelectMany(i => i).ToArray();
            Count = Data.Length;
            Shape = new int[] { data.Length, Count / data.Length };
        }

        public NDarray(NDarray<U> nd)
        {
            Shape = nd.Shape.ToArray();
            Count = nd.Count;
            Data = nd.Data.ToArray();
        }

        public override string ToString()
        {
            var reshape = $"reshape({Shape.Glue(",")})";
            var ndarray = $"np.array([{Data.Glue(",")}], dtype={OpsT.dtype})";
            return $"{ndarray}.{reshape}";
        }

        public U[] GetAtIndex(int idx)
        {
            var s = Utils.ArrMul(Shape, 1);
            return Data.Skip(idx * s).Take(s).ToArray();
        }

        public NDarray<V> Cast<V>()
        {
            var data = Data.Select(OpsT.Cast<V>).ToArray();
            return new NDarray<V>(data: data, shape: Shape);
        }

        public NDarray<U> Reshape(params int[] shape)
        {
            var nshape = Utils.PrepareReshape(Count, shape);
            return new NDarray<U>(data: Data.ToArray(), shape: nshape);
        }

        public NDarray<U> ReshapeInplace(params int[] shape)
        {
            Shape = Utils.PrepareReshape(Count, shape);
            return this;
        }

        public NDarray<U> Transpose(params int[] table)
        {
            if (table == null || table.Length == 0)
                table = Utils.PrepareTranspose(Shape.Length);

            if (table.Length != Shape.Length)
                throw new Exception();

            var strides = Utils.Shape2Strides(Shape);
            var nshape = Utils.DoTranspose(Shape, table);
            var nstrides = Utils.DoTranspose(strides, table);

            NDarray<U> nd0 = new NDarray<U>(nshape);
            for (int idx = 0; idx < Count; ++idx)
            {

                int idx1 = Utils.Int2IntIndex(idx, nshape, nstrides);
                nd0.Data[idx] = Data[idx1];
            }

            return nd0;
        }

        //public NDarray<U> T => Transpose();

        public void ApplyFuncInplace(Func<U, U> func)
        {
            for (int idx = 0; idx < Count; ++idx)
                Data[idx] = func(Data[idx]);
        }

        public void ApplyFuncInplace(Func<int, U, U> func)
        {
            for (int idx = 0; idx < Count; ++idx)
                Data[idx] = func(idx, Data[idx]);
        }

        public NDarray<U> ApplyFunc(Func<U, U> func)
        {
            NDarray<U> nd = new NDarray<U>(Shape);
            for (int idx = 0; idx < Count; ++idx)
                nd.Data[idx] = func(Data[idx]);

            return nd;
        }

        public (NDarray<U>, NDarray<U>) Split(int axis, int idx)
        {
            (var shape0, var shape1) = Utils.PrepareSplit(Shape, axis, idx);
            NDarray<U> nd0 = new NDarray<U>(shape0);
            NDarray<U> nd1 = new NDarray<U>(shape1);

            int[] indices = new int[Shape.Length];
            int[] strides = Utils.Shape2Strides(Shape);

            for (int idx0 = 0; idx0 < nd0.Count; ++idx0)
            {
                Utils.Int2ArrayIndex(idx0, nd0.Shape, indices);
                var idx2 = Utils.Array2IntIndex(indices, Shape, strides);
                nd0.Data[idx0] = Data[idx2];
            }

            for (int idx1 = 0; idx1 < nd1.Count; ++idx1)
            {
                Utils.Int2ArrayIndex(idx1, nd1.Shape, indices);
                indices[axis] += idx;
                var idx2 = Utils.Array2IntIndex(indices, Shape, strides);
                nd1.Data[idx1] = Data[idx2];
            }

            return (nd0, nd1);
        }

        public NDarray<U> Pad(params (int, int)[] pads)
        {
            (int[] nshape, var npads) = Utils.PreparePad(Shape, pads);
            var nd0 = new NDarray<U>(shape: nshape);

            int[] indices = new int[nshape.Length];
            int[] strides = Utils.Shape2Strides(nshape);
            for (int idx = 0; idx < Count; ++idx)
            {
                Utils.Int2ArrayIndex(idx, Shape, indices);
                for (int k = 0; k < indices.Length; ++k)
                {
                    (int f, int s) = npads[k];
                    indices[k] += f;
                }

                int idx0 = Utils.Array2IntIndex(indices, nd0.Shape, strides);
                nd0.Data[idx0] = Data[idx];
            }

            return nd0;
        }

    }
}
