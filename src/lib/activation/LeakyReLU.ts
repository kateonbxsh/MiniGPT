import ActivationFunction from "./ActivationFunction";

export default function LeakyReLU(alpha: number = 0.01) {
    
    return class extends ActivationFunction {

        override calculate(x: number): number {
            return x > 0 ? x : alpha * x;
        }
        override derivative(x: number): number {
            return x > 0 ? 1 : alpha;
        }

    }

}