import ActivationFunction from "./ActivationFunction";

export default class MinMaxNorm extends ActivationFunction {

    override calculate(x: number): number {
        const min = Math.min(...this.layer.neurons.map(neuron => neuron.value));
        const max = Math.max(...this.layer.neurons.map(neuron => neuron.value));
        return (x - min) / (max - min);
    }
    override derivative(x: number): number {
        const min = Math.min(...this.layer.neurons.map(neuron => neuron.value));
        const max = Math.max(...this.layer.neurons.map(neuron => neuron.value));
        return 1 / (max - min);
    }

}