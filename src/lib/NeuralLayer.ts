import NeuralNetwork from "./NeuralNetwork";
import Neuron from "./Neuron";
import NeuralFunction from "./abstract/NeuralFunction";

export default class NeuralLayer {

    network: NeuralNetwork;
    neurons: Array<Neuron> = []
    last: NeuralLayer | null = null;
    next: NeuralLayer | null = null;
    functions: Array<NeuralFunction> = [];

    constructor(neurons: number, ...outputFunctions: Array<{new(): NeuralFunction}>) {
        for(let i = 0; i < neurons; i++) {
            this.neurons.push(new Neuron(this));
        }
        this.functions = outputFunctions.map(neuralFunction => {
            let f = new neuralFunction();
            f.layer = this;
            return f;
        })
    }

    hasNext() {
        return this.next != null;
    }

    hasLast() {
        return this.last != null;
    }

    initiate() {
        this.neurons.forEach(neuron => neuron.initiate());
    }

    activate() {
        this.neurons.forEach((neuron, neuronNumber) => {
            if (!this.last) return;
            let sum = this.last.neurons.reduce((currentSum, currentNeuron) => {
                return currentSum + currentNeuron.value * currentNeuron.weights[neuronNumber];
            }, 0);
            sum += neuron.bias;
            neuron.value = sum;
        });
        if (this.next) {
            this.next.activate();
        }
    }

}