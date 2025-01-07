import ActivationFunction from "./ActivationFunction";

export default class Softmax extends ActivationFunction {

    override calculate(x: number): number {
        return Math.exp(x) / this.layer.neurons.map(neuron => Math.exp(neuron.value)).reduce((a, b) => a + b, 0);
    }
    override derivative(x: number): number {
        return this.calculate(x) * (1 - this.calculate(x));
    }

}