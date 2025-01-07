import NeuralLayer from "@/lib/models/components/NeuralLayer";
import MLPBuilder from "@/lib/models/mlp/MLPBuilder";
import NeuralFunction from "@/lib/activation/ActivationFunction";
import MLPTrainingSession from "@/lib/models/mlp/MLPTrainingSession";
import MultiLayerPerceptron from "./MultiLayerPerceptron";
import FFBBuilder from "./FFBBuilder";

export default class FeedForwardBlock extends MultiLayerPerceptron {

    inputs: number[][] = [];
    static override get Builder() {
        return new FFBBuilder();
    }

    resetInputs() {
        this.inputs = [];
    }

    backpropagateMultiple(outputGradient: number[][], learningRate: number) {
        return this.inputs.map((input, i) => {
            this.forward(input);
            this.backpropagate(outputGradient[i], learningRate);
            return this.layers[0].neurons.map(neuron => neuron.__cost_derivative);
        })
    }

}
