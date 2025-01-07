import MultiLayerPerceptron from "./MultiLayerPerceptron";
import NeuralLayer from "../components/NeuralLayer";

interface Gradient {
    weights: number[][][],
    biases: number[][]
}

export enum TrainingMethod {
    STOCHASTIC,
    BATCH,
    MINIBATCH //TO-DO
}

export default class MLPTrainingSession {

    network: MultiLayerPerceptron;
    gradients: Gradient[] = [];
    method: TrainingMethod = TrainingMethod.BATCH;
    learningRate = 1;

    constructor(network: MultiLayerPerceptron) {
        this.network = network;
    }

    setLearningRate(rate: number) {
        this.learningRate = rate;
        return this;
    }

    setMethod(method: TrainingMethod) {
        this.method = method;
        return this;
    }
    
    end() {
        if (this.method == TrainingMethod.STOCHASTIC) return;
        //average all gradients and subtract them from current weights and biases
        const n = this.gradients.length;
        let weightIndex = 0;
        for(let i = this.network.layers.length - 2; i >= 0; i--) {
            let currentLayer = this.network.layers[i];
            for(let j = 0; j < currentLayer.neurons.length; j++) {
                let currentNeuron = currentLayer.neurons[j];
                for(let k = 0; k < currentNeuron.weights.length; k++) {
                    let sum = 0;
                    for(let g = 0; g < n; g++) {
                        sum += this.gradients[g].weights[weightIndex][j][k];
                    }
                    currentNeuron.weights[k] -= this.learningRate * sum/n;
                }
            }
            weightIndex++;
        }
        let biasIndex = 0;
        for(let i = this.network.layers.length - 1; i >= 0; i--) {
            let currentLayer = this.network.layers[i];
            for(let j = 0; j < currentLayer.neurons.length; j++) {
                let currentNeuron = currentLayer.neurons[j];
                let sum = 0;
                for(let g = 0; g < n; g++) {
                    sum += this.gradients[g].biases[biasIndex][j];
                }
                currentNeuron.bias -= this.learningRate * sum/n;
            }
            biasIndex++;
        }
    }

}