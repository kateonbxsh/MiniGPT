import NeuralNetwork from "../NeuralNetwork";
import NeuralLayer from "../NeuralLayer";

interface NeuralGradient {
    weights: number[][][],
    biases: number[][]
}

export enum TrainingMethod {
    STOCHASTIC,
    BATCH,
    MINIBATCH //TO-DO
}

export default class NeuralNetworkTrainingSession {

    network: NeuralNetwork;
    gradients: NeuralGradient[] = [];
    method: TrainingMethod = TrainingMethod.BATCH;
    learningRate = 1;

    constructor(network: NeuralNetwork) {
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

    addData(input: Array<number>, desiredOutput: Array<number>) {
        this.network.activate(input);
        //back propagation
        const gradient: NeuralGradient = {
            weights: [],
            biases: []
        }
        let currentLayer: NeuralLayer | null = this.network.layers[this.network.layers.length - 1];
        currentLayer.neurons.forEach((neuron, i) => {
            neuron.__cost_derivative = 2 * (neuron.value - desiredOutput[i]);
        });
        while(currentLayer != null) {
            if (currentLayer.next) { // weights
                let layerWeightGradient = currentLayer.neurons.map(neuron => {
                    neuron.__cost_derivative = currentLayer!.next!.neurons.reduce((accumulator, otherNeuron, otherNeuronIndex) => {
                        return accumulator + neuron.weights[otherNeuronIndex] * otherNeuron.__cost_derivative;
                    }, 0);
                    return currentLayer!.next!.neurons.map((nextNeuron, nextNeuronIndex) => {
                        return nextNeuron.__cost_derivative * neuron.value;
                    })
                });
                gradient.weights.push(layerWeightGradient);
            }
            //biases
            let layerBiasGradient = currentLayer.neurons.map(neuron => {
                return neuron.__cost_derivative;
            });
            gradient.biases.push(layerBiasGradient)
            currentLayer = currentLayer.last;
        }
        if (this.method == TrainingMethod.BATCH) this.gradients.push(gradient);
        else {
            this.applyGradient(gradient);
        }
        return this;
    }

    applyGradient(gradient: NeuralGradient) {
        let weightIndex = 0;
        for(let i = this.network.layers.length - 2; i >= 0; i--) {
            let currentLayer = this.network.layers[i];
            for(let j = 0; j < currentLayer.neurons.length; j++) {
                let currentNeuron = currentLayer.neurons[j];
                for(let k = 0; k < currentNeuron.weights.length; k++) {
                    currentNeuron.weights[k] -= this.learningRate * gradient.weights[weightIndex][j][k];
                }
            }
            weightIndex++;
        }
        let biasIndex = 0;
        for(let i = this.network.layers.length - 1; i >= 0; i--) {
            let currentLayer = this.network.layers[i];
            for(let j = 0; j < currentLayer.neurons.length; j++) {
                let currentNeuron = currentLayer.neurons[j];
                currentNeuron.bias -= this.learningRate * gradient.biases[biasIndex][j];
            }
            biasIndex++;
        }
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