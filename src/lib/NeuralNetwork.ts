import NeuralLayer from "./NeuralLayer";
import NeuralNetworkBuilder from "./builder/NeuralNetworkBuilder";
import NeuralFunction from "./abstract/NeuralFunction";
import NeuralNetworkTrainingSession from "./builder/NeuralNetworkTrainingSession";

export default class NeuralNetwork {

    static get Builder() {
        return new NeuralNetworkBuilder();
    }

    layers: Array<NeuralLayer> = [];

    initiate() {
        this.layers.forEach(layer => layer.initiate());
    }

    toString() {
        return this.layers.map(layer => layer.neurons.map(neuron => neuron.value.toPrecision(2)).join(" ")).join("\n\n") + "\n\n";
    }

    activate(inputs: Array<number>) {
        if (inputs.length != this.layers[0].neurons.length) throw new Error("Input size needs to be the same as the input layer");
        this.layers[0].neurons.forEach((neuron, i) => {
            neuron.value = inputs[i];
        });
        this.layers[1].activate();
    }

    get output() {
        return this.layers[this.layers.length - 1].neurons.map(neuron => neuron.value);
    }

    cost(desiredOutput: Array<number>) {
        return this.output.reduce((accumulator, output, i) => accumulator += (desiredOutput[i] - output) ** 2, 0);
    }

    startTrainingSession() {
        return new NeuralNetworkTrainingSession(this);
    }

}
