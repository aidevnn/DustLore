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
