import MultiLayerPerceptron from "./MultiLayerPerceptron";
import NeuralLayer from "../components/NeuralLayer";
import ActivationFunction from "@/lib/activation/ActivationFunction";

export default class MLPBuilder {

    network: MultiLayerPerceptron = new MultiLayerPerceptron();
    hiddenLayers: Array<NeuralLayer> = [];
    input: number = 16;
    output: number = 16;
    outputFunction: typeof ActivationFunction | null = null;

    setInput(n: number) { this.input = n; return this;}
    setOutput(n: number, activationFunction: typeof ActivationFunction | null = null) { 
        this.output = n; 
        this.outputFunction = activationFunction; 
        return this; 
    }

    addLayer(layer: NeuralLayer, activationFunction: typeof ActivationFunction | null = null) {
        this.hiddenLayers.push(layer);
        if (activationFunction) layer.setFunction(activationFunction);
        return this;
    }

    build() {
        const outputLayer = new NeuralLayer(this.output);
        if (this.outputFunction) outputLayer.setFunction(this.outputFunction);
        this.network.layers = [
            new NeuralLayer(this.input),
            ...this.hiddenLayers,
            outputLayer
        ];
        let lastLayer: NeuralLayer | null = null;
        for(const layer of this.network.layers) {
            if (lastLayer) lastLayer.next = layer;
            layer.last = lastLayer;
            lastLayer = layer;
        }
        this.network.initiate();
        return this.network;
    }


}