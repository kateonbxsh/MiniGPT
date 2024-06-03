import NeuralNetwork from "../NeuralNetwork";
import NeuralLayer from "../NeuralLayer";
import NeuralFunction from "../abstract/NeuralFunction";

export default class NeuralNetworkBuilder {

    network: NeuralNetwork = new NeuralNetwork();
    hiddenLayers: Array<NeuralLayer> = [];
    input: number = 16;
    output: number = 16;

    setInput(n: number) { this.input = n; return this;}
    setOutput(n: number) { this.output = n; return this;}

    addLayer(layer: NeuralLayer) {
        this.hiddenLayers.push(layer);
        return this;
    }

    build() {
        this.network.layers = [
            new NeuralLayer(this.input),
            ...this.hiddenLayers,
            new NeuralLayer(this.output)
        ];
        let lastLayer: NeuralLayer | null = null;
        for(const layer of this.network.layers) {
            if (!(layer instanceof NeuralLayer)) continue;
            if (lastLayer) lastLayer.next = layer;
            layer.last = lastLayer;
            lastLayer = layer;
        }
        this.network.initiate();
        return this.network;
    }


}