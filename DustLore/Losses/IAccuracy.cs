using System;
using System.Linq;
using NDarrayLib;
namespace DustLore.Losses
{
    public interface IAccuracy
    {
        string Name { get; }
        double Func(NDarray<double> y, NDarray<double> p);
    }

    public class RoundAccuracy : IAccuracy
    {
        public string Name => "RoundAccuracy";

        public double Func(NDarray<double> y, NDarray<double> p)
        {
            int dim0 = y.Shape[0];
            int dim1 = y.Shape[1];
            double count = 0;
            for (int i = 0; i < dim0; ++i)
            {
                bool test = true;
                for(int j = 0; j < dim1; ++j)
                {
                    var yij = Math.Round(y.Data[i * dim1 + j]);
                    var pij = Math.Round(p.Data[i * dim1 + j]);
                    if (Math.Abs(yij - pij) > 1e-6)
                    {
                        test = false;
                        break;
                    }
                }
                count += test ? 1 : 0;
            }

            return count / dim0;
        }
    }

    public class ArgmaxAccuracy : IAccuracy
    {
        public string Name => "ArgmaxAccuracy";

        public double Func(NDarray<double> y, NDarray<double> p)
        {
            int dim0 = y.Shape[0];
            int dim1 = y.Shape[1];
            double count = 0;
            for(int i = 0; i < dim0; ++i)
            {
                var yi = Argmax(y.Data, i * dim1, dim1);
                var pi = Argmax(p.Data, i * dim1, dim1);
                count += yi == pi ? 1.0 : 0.0;
            }

            return count / dim0;
        }

        static int Argmax(double[] arr, int start, int length)
        {
            int bestIdx = 0;
            double bestVal = double.MinValue;
            for(int k = start; k < start + length; ++k)
            {
                var v = arr[k];
                if (v > bestVal)
                {
                    bestIdx = k - start;
                    bestVal = v;
                }
            }

            return bestIdx;
        }
    }
}
