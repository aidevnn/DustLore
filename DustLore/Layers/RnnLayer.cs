using System;
using System.Collections.Generic;
using System.Linq;
using DustLore.Activations;
using DustLore.Optimizers;
using NDarrayLib;

namespace DustLore.Layers
{
    public class RnnLayer : ILayer
    {
        public RnnLayer(int nUnits, int bttTrunc = 5, IActivation activation = null)
        {
            this.nUnits = nUnits;
            this.bttTrunc = bttTrunc;
            this.activation = activation ?? new TanhActivation();
        }

        public RnnLayer(int nUnits, (int, int) inputShape, int bttTrunc = 5, IActivation activation = null)
        {
            this.nUnits = nUnits;
            this.bttTrunc = bttTrunc;
            this.activation = activation ?? new TanhActivation();
            InputShape = new int[] { inputShape.Item1, inputShape.Item2 };
            OutputShape = InputShape.ToArray();
        }

        readonly int nUnits, bttTrunc;
        readonly IActivation activation;

        public string Name => "RnnLayer";

        public int Params => U.Count + V.Count + W.Count;

        public bool IsTraining { get; set; }
        public int[] InputShape { get; set; }
        public int[] OutputShape { get; set; }

        NDarray<double> U, V, W, lastInput, stateInput, states;
        IOptimizer uOpt, vOpt, wOpt;

        public NDarray<double> Backward(NDarray<double> accumGrad)
        {
            int timesteps = accumGrad.Shape[1];
            var gradU = new NDarray<double>(U.Shape);
            var gradV = new NDarray<double>(V.Shape);
            var gradW = new NDarray<double>(W.Shape);
            var accumGradNext = new NDarray<double>(accumGrad.Shape);

            for(int t = timesteps - 1; t >= 0; --t)
            {
                var accumGrad0 = GetArrAt(accumGrad, t);
                // grad_V += accum_grad[:, t].T.dot(self.states[:, t])
                gradV = ND.AddNDarray(ND.GemmTABC(accumGrad0, GetArrAt(states, t)), gradV);
                // grad_wrt_state = accum_grad[:, t].dot(self.V) * self.activation.gradient(self.state_input[:, t])
                var gradWrtState = ND.MulNDarray(ND.GemmABC(accumGrad0, V), activation.Grad(GetArrAt(stateInput, t)));
                // accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
                SetArrAt(t, accumGradNext, ND.GemmABC(gradWrtState, U));

                for(int t0 = t; t0 >= Math.Max(0, t - bttTrunc); --t0)
                {
                    // grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                    gradU = ND.AddNDarray(ND.GemmTABC(gradWrtState, GetArrAt(lastInput, t0)), gradU);
                    // grad_W += grad_wrt_state.T.dot(self.states[:, t_-1])
                    gradW = ND.AddNDarray(ND.GemmTABC(gradWrtState, GetArrAt(states, t0 - 1)), gradW);
                    // grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.gradient(self.state_input[:, t_-1])
                    gradWrtState = ND.MulNDarray(ND.GemmABC(gradWrtState, W), activation.Grad(GetArrAt(stateInput, t0 - 1)));
                }
            }

            if (IsTraining)
            {
                uOpt.Update(U, gradU);
                vOpt.Update(V, gradV);
                wOpt.Update(W, gradW);
            }

            return accumGradNext;
        }

        public NDarray<double> Forward(NDarray<double> X, bool isTraining)
        {
            lastInput = new NDarray<double>(X);
            IsTraining = isTraining;
            int batchSize = X.Shape[0], timeSteps = X.Shape[1], inputDim = X.Shape[2];

            stateInput = new NDarray<double>(batchSize, timeSteps, nUnits);
            states = new NDarray<double>(batchSize, timeSteps + 1, nUnits);
            var outputs = new NDarray<double>(batchSize, timeSteps, inputDim);

            for(int t = 0; t < timeSteps; ++t)
            {
                // self.state_input[:, t] = X[:, t].dot(self.U.T) + self.states[:, t-1].dot(self.W.T)
                SetArrAt(t, stateInput, ND.AddNDarray(ND.GemmATBC(GetArrAt(X, t), U), ND.GemmATBC(GetArrAt(states, t - 1), W)));
                // self.states[:, t] = self.activation(self.state_input[:, t])
                SetArrAt(t, states, activation.Func(GetArrAt(stateInput, t)));
                // self.outputs[:, t] = self.states[:, t].dot(self.V.T)
                SetArrAt(t, outputs, ND.GemmATBC(GetArrAt(states, t), V));
            }

            return outputs;
        }

        public int[] GetOutputShape() => OutputShape;

        public void Initialize(IOptimizer optimizer)
        {
            int timesteps = InputShape[0], inputDim = InputShape[1];
            double lim0 = 1.0 / Math.Sqrt(inputDim);
            double lim1 = 1.0 / Math.Sqrt(nUnits);
            U = ND.Uniform(-lim0, lim0, nUnits, inputDim);
            V = ND.Uniform(-lim1, lim1, inputDim, nUnits);
            W = ND.Uniform(-lim1, lim1, nUnits, nUnits);

            uOpt = optimizer.Clone();
            vOpt = optimizer.Clone();
            wOpt = optimizer.Clone();
        }

        public void SetInputShape(int[] shape)
        {
            InputShape = shape.ToArray();
            OutputShape = shape.ToArray();
        }

        // GetSubArray a[:, t]; a.Shape length must be 3
        public static NDarray<double> GetArrAt(NDarray<double> a, int t)
        {
            if (a.Shape.Length != 3)
                throw new Exception();

            int s0 = a.Shape[0], s1 = a.Shape[1], s2 = a.Shape[2];
            int s12 = s1 * s2;
            t = (t + s1) % s1;

            var nd = new NDarray<double>(s0, s2);
            for (int i = 0; i < s0; ++i)
            {
                for (int j = 0; j < s2; ++j)
                {
                    int idx0 = i * s12 + t * s2 + j;
                    int idx1 = i * s2 + j;
                    nd.Data[idx1] = a.Data[idx0];
                }
            }

            return nd;
        }

        // SetArrAt a[:, t] = sub; a.Shape.length = 3 and b.Shape.Length = 2
        public static void SetArrAt(int t, NDarray<double> a, NDarray<double> sub)
        {
            int s0 = a.Shape[0], s1 = a.Shape[1], s2 = a.Shape[2];
            int s12 = s1 * s2;
            t = (t + s1) % s1;

            if (sub.Shape[0] != s0 || sub.Shape[1] != s2)
                throw new Exception();

            for (int i = 0; i < s0; ++i)
            {
                for (int j = 0; j < s2; ++j)
                {
                    int idx0 = i * s12 + t * s2 + j;
                    int idx1 = i * s2 + j;
                    a.Data[idx0] = sub.Data[idx1];
                }
            }
        }

    }
}
