import Matrix from "@/lib/math/Matrix";

export default class AttentionBlock<TokenSize extends number = number> {
    
    tokenSize: TokenSize;
    weightsQ: Matrix<TokenSize, TokenSize>;
    weightsK: Matrix<TokenSize, TokenSize>;
    weightsV: Matrix<TokenSize, TokenSize>;
    outputWeights: Matrix<TokenSize, TokenSize>;
    input: Matrix<number, TokenSize>;

    learningT: number = 1;
    MweightsQ: Matrix<TokenSize, TokenSize>;
    MweightsK: Matrix<TokenSize, TokenSize>;
    MweightsV: Matrix<TokenSize, TokenSize>;
    MoutputWeights: Matrix<TokenSize, TokenSize>;
    VweightsQ: Matrix<TokenSize, TokenSize>;
    VweightsK: Matrix<TokenSize, TokenSize>;
    VweightsV: Matrix<TokenSize, TokenSize>;
    VoutputWeights: Matrix<TokenSize, TokenSize>;

    BETA1: number = 0.9;
    BETA2: number = 0.999;
    EPSILON: number = 1e-8;

    constructor(tokenSize: TokenSize) {

        this.tokenSize = tokenSize;
        const alpha = Math.sqrt(1/tokenSize);
        this.weightsQ = new Matrix(tokenSize, tokenSize, () => Math.random() * alpha - alpha/2);
        this.weightsK = new Matrix(tokenSize, tokenSize, () => Math.random() * alpha - alpha/2);
        this.weightsV = new Matrix(tokenSize, tokenSize, () => Math.random() * alpha - alpha/2);
        this.outputWeights = new Matrix(tokenSize, tokenSize, () => Math.random() * alpha - alpha/2);
        this.MweightsQ = new Matrix(tokenSize, tokenSize, () => 0);
        this.MweightsK = new Matrix(tokenSize, tokenSize, () => 0);
        this.MweightsV = new Matrix(tokenSize, tokenSize, () => 0);
        this.MoutputWeights = new Matrix(tokenSize, tokenSize, () => 0);
        this.VweightsQ = new Matrix(tokenSize, tokenSize, () => 0);
        this.VweightsK = new Matrix(tokenSize, tokenSize, () => 0);
        this.VweightsV = new Matrix(tokenSize, tokenSize, () => 0);
        this.VoutputWeights = new Matrix(tokenSize, tokenSize, () => 0);

    }

    forward(input: Matrix<number, TokenSize>): Matrix {
        this.input = input;

        // Compute Q, K, V
        const Q = Matrix.multiply(input, this.weightsQ);
        const K = Matrix.multiply(input, this.weightsK);
        const V = Matrix.multiply(input, this.weightsV);

        // Compute attention scores
        const sqrtD = Math.sqrt(this.tokenSize);
        const scores = Matrix.softmax(Matrix.multiplyScalar(Matrix.multiply(Q, K.transpose()), sqrtD));

        // Compute weighted sum
        const weightedSum = Matrix.multiply(scores, V);

        // Apply output weights
        const attentionOutput = Matrix.multiply(weightedSum, this.outputWeights);

        // Add residual connection
        const output = Matrix.add(input, attentionOutput); // Residual connection

        return output; // No LayerNorm applied in this implementation
    }

    backpropagate(gradient: number[][], learningRate: number): Matrix {

        const input = this.input;
        const outputGradient = new Matrix(this.input.rows, this.tokenSize, gradient);

        // Split the gradient: part goes to attention, part goes to residual
        const attentionGradient = outputGradient; // Gradients for attention output
        const residualGradient = outputGradient; // Gradients for the residual connection (identity)

        const Q = Matrix.multiply(input, this.weightsQ);
        const K = Matrix.multiply(input, this.weightsK);
        const V = Matrix.multiply(input, this.weightsV);

        // Attention gradients
        const sqrtD = Math.sqrt(this.tokenSize);
        const scores = Matrix.softmax(Matrix.multiplyScalar(Matrix.multiply(Q, K.transpose()), sqrtD));
        
        const weightedSum = Matrix.multiply(scores, V);

        const dOutputWeights = Matrix.multiply(weightedSum.transpose(), attentionGradient);
        const dWeightedSum = Matrix.multiply(attentionGradient, this.outputWeights.transpose());

        const dScores = Matrix.multiply(dWeightedSum, V.transpose());
        const dV = Matrix.multiply(scores.transpose(), dWeightedSum);

        const dQ = Matrix.multiply(dScores, K);
        const dK = Matrix.multiply(dScores.transpose(), Q);

        const dWeightsQ = Matrix.multiply(input.transpose(), dQ);
        const dWeightsK = Matrix.multiply(input.transpose(), dK);
        const dWeightsV = Matrix.multiply(input.transpose(), dV);

        // Update weights with Adam
        this.updateAdam(dWeightsQ, dWeightsK, dWeightsV, dOutputWeights, learningRate);

        // Backpropagate to input
        const inputGradient = Matrix.multiply(dQ, this.weightsQ.transpose())
            .add(Matrix.multiply(dK, this.weightsK.transpose()))
            .add(Matrix.multiply(dV, this.weightsV.transpose()));

        // Combine gradients from attention and residual paths
        const finalInputGradient = Matrix.add(inputGradient, residualGradient);

        this.learningT++;
        return finalInputGradient;
    }

    private updateAdam(
        dWeightsQ: Matrix<TokenSize, TokenSize>,
        dWeightsK: Matrix<TokenSize, TokenSize>,
        dWeightsV: Matrix<TokenSize, TokenSize>,
        dOutputWeights: Matrix<TokenSize, TokenSize>,
        learningRate: number
    ): void {
        // Adam optimizer update logic (unchanged, modularized for clarity)

        this.MweightsQ = Matrix.subtract(
            Matrix.multiplyScalar(this.MweightsQ, this.BETA1),
            Matrix.multiplyScalar(dWeightsQ, 1 - this.BETA1)
        );
        this.MweightsK = Matrix.subtract(
            Matrix.multiplyScalar(this.MweightsK, this.BETA1),
            Matrix.multiplyScalar(dWeightsK, 1 - this.BETA1)
        );
        this.MweightsV = Matrix.subtract(
            Matrix.multiplyScalar(this.MweightsV, this.BETA1),
            Matrix.multiplyScalar(dWeightsV, 1 - this.BETA1)
        );
        this.MoutputWeights = Matrix.subtract(
            Matrix.multiplyScalar(this.MoutputWeights, this.BETA1),
            Matrix.multiplyScalar(dOutputWeights, 1 - this.BETA1)
        );

        this.VweightsQ = Matrix.add(
            Matrix.multiplyScalar(this.VweightsQ, this.BETA2),
            Matrix.multiplyScalar(dWeightsQ.map((v) => v * v), 1 - this.BETA2)
        );
        this.VweightsK = Matrix.add(
            Matrix.multiplyScalar(this.VweightsK, this.BETA2),
            Matrix.multiplyScalar(dWeightsK.map((v) => v * v), 1 - this.BETA2)
        );
        this.VweightsV = Matrix.add(
            Matrix.multiplyScalar(this.VweightsV, this.BETA2),
            Matrix.multiplyScalar(dWeightsV.map((v) => v * v), 1 - this.BETA2)
        );
        this.VoutputWeights = Matrix.add(
            Matrix.multiplyScalar(this.VoutputWeights, this.BETA2),
            Matrix.multiplyScalar(dOutputWeights.map((v) => v * v), 1 - this.BETA2)
        );

        const correctedMweightsQ = this.MweightsQ.map(
            (v) => v / (1 - Math.pow(this.BETA1, this.learningT))
        );
        const correctedMweightsK = this.MweightsK.map(
            (v) => v / (1 - Math.pow(this.BETA1, this.learningT))
        );
        const correctedMweightsV = this.MweightsV.map(
            (v) => v / (1 - Math.pow(this.BETA1, this.learningT))
        );
        const correctedMoutputWeights = this.MoutputWeights.map(
            (v) => v / (1 - Math.pow(this.BETA1, this.learningT))
        );

        const correctedVweightsQ = this.VweightsQ.map(
            (v) => v / (1 - Math.pow(this.BETA2, this.learningT))
        );
        const correctedVweightsK = this.VweightsK.map(
            (v) => v / (1 - Math.pow(this.BETA2, this.learningT))
        );
        const correctedVweightsV = this.VweightsV.map(
            (v) => v / (1 - Math.pow(this.BETA2, this.learningT))
        );
        const correctedVoutputWeights = this.VoutputWeights.map(
            (v) => v / (1 - Math.pow(this.BETA2, this.learningT))
        );

        this.weightsQ = Matrix.subtract(
            this.weightsQ,
            correctedMweightsQ.map((M, i, j) => M * learningRate / (Math.sqrt(correctedVweightsQ.get(i, j)) + this.EPSILON))
        );
        this.weightsK = Matrix.subtract(
            this.weightsK,
            correctedMweightsK.map((M, i, j) => M * learningRate / (Math.sqrt(correctedVweightsK.get(i, j)) + this.EPSILON))
        );
        this.weightsV = Matrix.subtract(
            this.weightsV,
            correctedMweightsV.map((M, i, j) => M * learningRate / (Math.sqrt(correctedVweightsV.get(i, j)) + this.EPSILON))
        );
        this.outputWeights = Matrix.subtract(
            this.outputWeights,
            correctedMoutputWeights.map((M, i, j) => M * learningRate / (Math.sqrt(correctedVoutputWeights.get(i, j)) + this.EPSILON))
        );

    }

}