import NeuralLayer from "@/lib/models/components/NeuralLayer";
import MLPBuilder from "@/lib/models/mlp/MLPBuilder";
import NeuralFunction from "@/lib/activation/ActivationFunction";
import MLPTrainingSession from "@/lib/models/mlp/MLPTrainingSession";

interface Gradient {
    weights: number[][][],
    biases: number[][]
}


export default class MultiLayerPerceptron {

    learningT: number = 1;
    BETA1: number = 0.9;
    BETA2: number = 0.999;
    EPSILON: number = 1e-8;
    MGradient: Gradient;
    VGradient: Gradient;

    static get Builder() {
        return new MLPBuilder();
    }

    layers: Array<NeuralLayer> = [];

    initiate() {
        this.layers.forEach(layer => layer.initiate());
        this.MGradient = {
            weights: this.layers.slice(0, this.layers.length - 1).map(layer => layer.neurons.map(n => n.weights.map(() => 0))),
            biases: this.layers.map(layer => layer.neurons.map(n => 0))
        }
        this.VGradient = {
            weights: this.layers.slice(0, this.layers.length - 1).map(layer => layer.neurons.map(n => n.weights.map(() => 0))),
            biases: this.layers.map(layer => layer.neurons.map(n => 0))
        }
    }

    toString() {
        return this.layers.map(layer => layer.neurons.map(neuron => neuron.value.toPrecision(2)).join(" ")).join("\n\n") + "\n\n";
    }

    forward(inputs: Array<number>) {
        if (inputs.length != this.layers[0].neurons.length) throw new Error("Input size needs to be the same as the input layer");
        this.layers[0].neurons.forEach((neuron, i) => {
            neuron.value = inputs[i];
            neuron.valueAfterFunc = inputs[i];
        });
        this.layers[1].forward();
    }

    get output() {
        return this.layers[this.layers.length - 1].neurons.map(neuron => neuron.valueAfterFunc);
    }

    cost(desiredOutput: Array<number>) {
        return this.output.reduce((accumulator, output, i) => accumulator += (desiredOutput[i] - output) ** 2, 0);
    }

    startTrainingSession() {
        return new MLPTrainingSession(this);
    }

    backpropagate(outputGradient: Array<number>, learningRate: number) {
        
        const gradient: Gradient = {
            weights: [],
            biases: []
        }
        let currentLayer: NeuralLayer | null = this.layers[this.layers.length - 1];
        currentLayer.neurons.forEach((neuron, i) => {
            neuron.__cost_derivative = outputGradient[i];
        });
        while(currentLayer != null) {
            if (currentLayer.next) { // weights
                let layerWeightGradient = currentLayer.neurons.map(neuron => {
                    const nextHasFunction = currentLayer!.next!.hasFunction();
                    neuron.__cost_derivative = currentLayer!.next!.neurons.reduce((accumulator, otherNeuron, otherNeuronIndex) => {
                        const functionDerivative = nextHasFunction ? currentLayer!.next!.function!.derivative(otherNeuron.value) : 1;
                        return accumulator + neuron.weights[otherNeuronIndex] * functionDerivative * otherNeuron.__cost_derivative;
                    }, 0);
                    return currentLayer!.next!.neurons.map((nextNeuron, nextNeuronIndex) => {
                        const functionDerivative = nextHasFunction ? currentLayer!.next!.function!.derivative(nextNeuron.value) : 1;
                        return nextNeuron.__cost_derivative * functionDerivative * neuron.value;
                    })
                });
                gradient.weights.unshift(layerWeightGradient);
            }
            //biases
            let layerBiasGradient = currentLayer.neurons.map(neuron => {
                return neuron.__cost_derivative;
            });
            gradient.biases.unshift(layerBiasGradient)
            currentLayer = currentLayer.last;
        }
        
        //ADAM
        this.MGradient.weights = this.MGradient.weights.map((layer, i) => 
            layer.map((weightList, j) => 
                weightList.map((weight, k) => {
                    return weight * this.BETA1 + gradient.weights[i][j][k] * (1 - this.BETA1);
                })
            )
        );
        this.VGradient.weights = this.VGradient.weights.map((layer, i) => 
            layer.map((weightList, j) => 
                weightList.map((weight, k) => {
                    return weight * this.BETA2 + Math.pow(gradient.weights[i][j][k], 2) * (1 - this.BETA2);
                })
            )
        );

        const correctedMGradientWeights = this.MGradient.weights.map((layer, i) => 
            layer.map((weightList, j) => 
                weightList.map((m, k) => {
                    return m / (1 - Math.pow(this.BETA1, this.learningT));
                })
            )
        );
        const correctedVGradientWeights = this.VGradient.weights.map((layer, i) => 
            layer.map((weightList, j) => 
                weightList.map((v, k) => {
                    return v / (1 - Math.pow(this.BETA2, this.learningT));
                })
            )
        );

        for(let i = 0; i < this.layers.length - 1; i++) {
            let currentLayer = this.layers[i];
            for(let j = 0; j < currentLayer.neurons.length; j++) {
                let currentNeuron = currentLayer.neurons[j];
                for(let k = 0; k < currentNeuron.weights.length; k++) {
                    currentNeuron.weights[k] -= 
                    learningRate * correctedMGradientWeights[i][j][k] / 
                    (Math.sqrt(correctedVGradientWeights[i][j][k]) + this.EPSILON);
                }
            }
        }

        this.MGradient.biases = this.MGradient.biases.map((layer, i) => 
            layer.map((bias, j) => {
                return bias * this.BETA2 + gradient.biases[i][j] * (1 - this.BETA2);
            })
        );
        this.VGradient.biases = this.VGradient.biases.map((layer, i) => 
            layer.map((bias, j) => {
                return bias * this.BETA2 + gradient.biases[i][j] * gradient.biases[i][j] * (1 - this.BETA2);
            })
        );

        const correctedMGradientBiases = this.MGradient.biases.map((layer, i) => 
            layer.map((m, j) => {
                return m / (1 - Math.pow(this.BETA1, this.learningT));
            })
        );
        const correctedVGradientBiases = this.VGradient.biases.map((layer, i) => 
            layer.map((v, j) => {
                return v / (1 - Math.pow(this.BETA2, this.learningT));
            })
        );

        for(let i = 0; i < this.layers.length; i++) {
            let currentLayer = this.layers[i];
            for(let j = 0; j < currentLayer.neurons.length; j++) {
                let currentNeuron = currentLayer.neurons[j];
                currentNeuron.bias -= learningRate * correctedMGradientBiases[i][j] / 
                (Math.sqrt(correctedVGradientBiases[i][j]) + this.EPSILON);
            }
        }

        this.learningT++;

    }

}
