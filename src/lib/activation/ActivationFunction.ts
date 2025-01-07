import NeuralLayer from "../models/components/NeuralLayer";

export default class ActivationFunction {

    layer: NeuralLayer;

    constructor(layer: NeuralLayer) {
        this.layer = layer;
    }

    calculate(x: number): number { return x; };
    derivative(x: number): number { return 1; };

}