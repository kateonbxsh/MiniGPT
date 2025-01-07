import Matrix from "@/lib/math/Matrix";

export default class LayerNorm<TokenSize extends number = number>
 {

    featureSize: TokenSize;
    epsilon: number;
    gamma: Matrix<1, TokenSize>;
    beta: Matrix<1, TokenSize>;
    input: Matrix<number, TokenSize>; // Store the input for backpropagation
    mean: Matrix<1, TokenSize>; // Store the mean for backpropagation
    variance: Matrix<1, TokenSize>; // Store the variance for backpropagation

    mGamma: Matrix<1, TokenSize>; // First moment (m) for gamma
    vGamma: Matrix<1, TokenSize>; // Second moment (v) for gamma
    mBeta: Matrix<1, TokenSize>; // First moment (m) for beta
    vBeta: Matrix<1, TokenSize>; // Second moment (v) for beta
    beta1: number = 0.9; // Exponential decay rate for the first moment estimate
    beta2: number = 0.999; // Exponential decay rate for the second moment estimate
    epsilonAdam: number = 1e-5; // Small value to avoid division by zero

    timestep: number = 0; // To track the timestep for bias correction

    constructor(featureSize: TokenSize, epsilon: number = 1e-5) {
        this.epsilon = epsilon;
        this.featureSize = featureSize;

        // Initialize gamma and beta (learnable parameters)
        this.gamma = new Matrix(1, featureSize, () => 1); // Scale initialized to 1
        this.beta = new Matrix(1, featureSize, () => 0); // Shift initialized to 0
        this.mGamma = new Matrix(1, featureSize, () => 0);
        this.vGamma = new Matrix(1, featureSize, () => 0);
        this.mBeta = new Matrix(1, featureSize, () => 0);
        this.vBeta = new Matrix(1, featureSize, () => 0);
    }

    /**
     * Forward pass of LayerNorm
     * @param input - The input matrix to normalize
     * @returns Normalized output matrix
     */
    forward(input: Matrix<number, TokenSize>): Matrix<number, TokenSize> {
        
        this.input = input;

        this.mean = input.mean('col');
        this.variance = input.variance('col');

        // Normalize input: (x - mean) / sqrt(variance + epsilon)
        const normalized = input.map(
            (val, i, j) => (val - this.mean.get(0, j)) / (Math.sqrt(this.variance.get(0, j) + this.epsilon))
        );

        // Apply scale (gamma) and shift (beta)
        const output = normalized.map(
            (val, i, j) => val * this.gamma.get(0, j) + this.beta.get(0, j)
        );

        return output;
    }

    /**
     * Backpropagation for LayerNorm
     * @param gradOutput - Gradient of the loss with respect to the output
     * @param learningRate - Learning rate for parameter updates
     * @returns Gradient of the loss with respect to the input
     */
    backpropagate(outputGradient: number[][], learningRate: number): Matrix {

        const gradOutput = new Matrix(this.input.rows, this.featureSize, outputGradient);
        const gradNormalized = gradOutput.map((val, i, j) => val * this.gamma.get(0, j));

        // Gradients w.r.t. gamma and beta
        const gradGamma = gradNormalized.sum('col');
        const gradBeta = gradOutput.sum('col');

        // Update the moment estimates for gamma
        this.mGamma = this.mGamma.multiplyScalar(this.beta1).add(gradGamma.multiplyScalar(1 - this.beta1));
        this.vGamma = this.vGamma.multiplyScalar(this.beta2).add(gradGamma.map((val) => val * val).multiplyScalar(1 - this.beta2));

        // Update the moment estimates for beta
        this.mBeta = this.mBeta.multiplyScalar(this.beta1).add(gradBeta.multiplyScalar(1 - this.beta1));
        this.vBeta = this.vBeta.multiplyScalar(this.beta2).add(gradBeta.map((val) => val * val).multiplyScalar(1 - this.beta2));

        // Bias correction
        const mGammaCorrected = this.mGamma.map((val) => val / (1 - Math.pow(this.beta1, this.timestep + 1)));
        const vGammaCorrected = this.vGamma.map((val) => val / (1 - Math.pow(this.beta2, this.timestep + 1)));
        const mBetaCorrected = this.mBeta.map((val) => val / (1 - Math.pow(this.beta1, this.timestep + 1)));
        const vBetaCorrected = this.vBeta.map((val) => val / (1 - Math.pow(this.beta2, this.timestep + 1)));

        const gammaUpdate = mGammaCorrected.multiplyScalar(learningRate).divide(vGammaCorrected.map((val) => Math.sqrt(val) + this.epsilonAdam));
        const betaUpdate = mBetaCorrected.multiplyScalar(learningRate).divide(vBetaCorrected.map((val) => Math.sqrt(val) + this.epsilonAdam));

        this.gamma = this.gamma.subtract(gammaUpdate);
        this.beta = this.beta.subtract(betaUpdate);

        const gradInput = gradNormalized.map((val, i, j) => val / Math.sqrt(this.variance.get(0, j) + this.epsilon));

        this.timestep++;
        

        return gradInput;
    }
}
