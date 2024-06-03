import NeuralLayer from "../NeuralLayer";

export default abstract class NeuralFunction {

    layer: NeuralLayer;

    abstract calculate(x: number): number;
    abstract derivative(x: number): number;

}