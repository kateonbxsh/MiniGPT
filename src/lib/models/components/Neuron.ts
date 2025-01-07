import NeuralLayer from "./NeuralLayer";

export default class Neuron {

    layer: NeuralLayer;
    value: number = 0;
    valueAfterFunc: number = 0;
    weights: Array<number> = [];
    bias: number = 0;

    __cost_derivative: number = 0;

    constructor(layer: NeuralLayer) {this.layer = layer;}

    initiate() {
        if (this.layer.next) this.weights = this.layer.next.neurons.map(neuron => Math.random() * 0.1 - 0.5);
    }

}