import ActivationFunction from "./ActivationFunction";

export default class ReLU extends ActivationFunction {

    override calculate(x: number): number {
        return x > 0 ? x : 0;
    }
    override derivative(x: number): number {
        return x > 0 ? 1 : 0;
    }

}