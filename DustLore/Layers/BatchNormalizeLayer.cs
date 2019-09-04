using System;
using System.Linq;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public class BatchNormalizeLayer : ILayer
    {
        public BatchNormalizeLayer(double momentum = 0.99)
        {
            this.momentum = momentum;
        }

        readonly double momentum;
        public string Name => "BatchNormalizeLayer";

        public int Params => gamma.Count + beta.Count;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        NDarray<double> gamma, beta, runningMean, runningVar, xCentered, stdDevInv, xNorm;
        IOptimizer bOpt, gOpt;

        public NDarray<double> Backward(NDarray<double> accumGrad)
        {
            var gamma0 = new NDarray<double>(gamma);

            var s0 = ND.SumAxis(accumGrad, 0);
            var s1 = ND.SumAxis(ND.MulNDarray(accumGrad, xCentered), 0);

            if (IsTraining)
            {
                var gGamma = ND.SumAxis(ND.MulNDarray(accumGrad, xNorm), 0);
                var gBeta = s0;

                gOpt.Update(gamma, gGamma);
                bOpt.Update(beta, gBeta);
            }

            double batchSize = accumGrad.Shape[0];
            var m0 = ND.MulNDarray(gamma0, stdDevInv, 1.0 / batchSize, 1);
            var m1 = ND.MulNDarray(xNorm, ND.MulNDarray(stdDevInv, s1));
            var diff = ND.SubNDarray(accumGrad, ND.AddNDarray(s0, m1), batchSize, 1);
            accumGrad = ND.MulNDarray(m0, diff);

            return accumGrad;
        }

        public NDarray<double> Forward(NDarray<double> X, bool isTraining)
        {
            IsTraining = isTraining;
            if (runningMean == null)
            {
                runningMean = ND.MeanAxis(X, 0);
                runningVar = ND.VarAxis(X, 0);
            }

            NDarray<double> mean = null, var = null;
            if (isTraining)
            {
                mean = ND.MeanAxis(X, 0);
                var = ND.VarAxis(X, 0);
                runningMean = ND.AddNDarray(runningMean, mean, momentum, 1 - momentum);
                runningVar = ND.AddNDarray(runningVar, var, momentum, 1 - momentum);
            }
            else
            {
                mean = runningMean;
                var = runningVar;
            }

            xCentered = ND.SubNDarray(X, mean);
            stdDevInv = var.ApplyFunc(x => 1.0 / Math.Sqrt(x + 0.01));

            xNorm = ND.MulNDarray(xCentered, stdDevInv);
            var output = ND.MulNDarray(gamma, xNorm);
            output = ND.AddNDarray(output, beta);
            return output;
        }

        public int[] GetOutputShape() => OutputShape;

        public void Initialize(IOptimizer optimizer)
        {
            gamma = new NDarray<double>(1.0, InputShape);
            beta = new NDarray<double>(0.0, InputShape);
            gOpt = optimizer.Clone();
            bOpt = optimizer.Clone();
        }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
            OutputShape = shape.ToArray();
        }

        public void ImportWeights(string w, string b)
        {

        }
    }
}
