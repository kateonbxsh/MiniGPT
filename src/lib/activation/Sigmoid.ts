import ActivationFunction from "./ActivationFunction";

export default class Sigmoid extends ActivationFunction {

    override calculate(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }
    override derivative(x: number): number {
        return this.calculate(x) * this.calculate(1 - x);
    }

}