import MultiLayerPerceptron from "@/lib/models/mlp/MultiLayerPerceptron";
import Neuron from "@/lib/models/components/Neuron";
import ActivationFunction from "@/lib/activation/ActivationFunction";

export default class NeuralLayer {

    network: MultiLayerPerceptron;
    neurons: Array<Neuron> = []
    last: NeuralLayer | null = null;
    next: NeuralLayer | null = null;
    function: ActivationFunction | null = null;

    constructor(neurons: number) {
        for(let i = 0; i < neurons; i++) {
            this.neurons.push(new Neuron(this));
        }
    }

    setFunction(func: typeof ActivationFunction) {
        this.function = new func(this);
        return this;
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

    hasFunction() {
        return this.function != null;
    }

    forward() {
        this.neurons.forEach((neuron, neuronNumber) => {
            if (!this.last) return;
            let sum = this.last.neurons.reduce((currentSum, currentNeuron) => {
                return currentSum + currentNeuron.valueAfterFunc * currentNeuron.weights[neuronNumber];
            }, 0);
            sum += neuron.bias;
            neuron.value = sum;
            neuron.valueAfterFunc = sum;
        });
        if(this.hasFunction()) {
            this.neurons.forEach((neuron, i) => {neuron.valueAfterFunc = this.function!.calculate(neuron.value);});
        }
        if (this.next) {
            this.next.forward();
        }
    }

}