using System;
namespace NDarrayLib
{
    public static class NDsharp
    {

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

            for (int i = 0; i < wa; ++i)
            {
                for (int j = 0; j < hb; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < ha; ++k)
                        sum += a.Data[i * ha + k] * b.Data[k * hb + j];

                    nd.Data[i * hb + j] += sum;
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

            if (ha != hb || (c != null && (c.Count != wa && c.Count != wb && c.Count != wa * wb)))
                throw new Exception();

            var nd = new NDarray<double>(wa, wb);
            if (c != null)
                nd = ND.AddNDarray(nd, c);

            for (int i = 0; i < wa; ++i)
            {
                for (int j = 0; j < wb; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < ha; ++k)
                        sum += a.Data[i * ha + k] * b.Data[j * ha + k];

                    nd.Data[i * wb + j] += sum;
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

            if (wa != wb || (c != null && (c.Count != ha && c.Count != hb && c.Count != ha * hb)))
                throw new Exception();

            var nd = new NDarray<double>(ha, hb);
            if (c != null)
                nd = ND.AddNDarray(nd, c);

            for (int i = 0; i < ha; ++i)
            {
                for (int j = 0; j < hb; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < wa; ++k)
                        sum += a.Data[k * ha + i] * b.Data[k * hb + j];

                    nd.Data[i * hb + j] += sum;
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

            if (wa != hb || (c != null && (c.Count != ha && c.Count != wb && c.Count != ha * wb)))
                throw new Exception();

            var nd = new NDarray<double>(ha, wb);
            if (c != null)
                nd = ND.AddNDarray(nd, c);

            for (int i = 0; i < ha; ++i)
            {
                for (int j = 0; j < wb; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < wa; ++k)
                        sum += a.Data[k * ha + i] * b.Data[j * wa + k];

                    nd.Data[i * wb + j] += sum;
                }
            }

            return nd;
        }

    }
}
