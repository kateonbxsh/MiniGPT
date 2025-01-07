import ActivationFunction from "./ActivationFunction";

export default class GELU extends ActivationFunction {

    override calculate(x: number): number {
        // GELU formula: x * 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        const coefficient = Math.sqrt(2 / Math.PI);
        return 0.5 * x * (1 + Math.tanh(coefficient * (x + 0.044715 * Math.pow(x, 3))));
    }

    override derivative(x: number): number {
        // Derivative of GELU: 
        // d/dx of GELU(x) = GELU(x) + x * (1 - GELU(x)^2) * coefficient
        const gelu = this.calculate(x);
        return gelu + x * (1 - Math.pow(gelu, 2)) * Math.sqrt(2 / Math.PI);
    }
}
