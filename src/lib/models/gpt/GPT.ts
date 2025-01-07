import Matrix from "@/lib/math/Matrix";
import AttentionBlock from "../components/AttentionBlock";
import MultiLayerPerceptron from "@/lib/models/mlp/MultiLayerPerceptron";
import Softmax from "@/lib/activation/Softmax";
import natural from 'natural';
import LayerNorm from "../components/LayerNorm";
import FeedForwardBlock from "../mlp/FeedForwardBlock";

export default class GPT {

    blocks: Array<AttentionBlock | FeedForwardBlock | LayerNorm> = [];
    dictionary: Map<string, number> = new Map();
    reverseDictionary: Map<number, string> = new Map();

    vocabularySize: number = 1000;
    dimensionality: number;

    BETA1: number = 0.9;
    BETA2: number = 0.999;
    EPSILON: number = 1e-8;
    learningT: number = 1;

    vocabulary: Matrix = new Matrix(0, 0);
    vocabularyMGradient: Matrix = new Matrix(0, 0);
    vocabularyVGradient: Matrix = new Matrix(0, 0);

    finalFFB: FeedForwardBlock;
    learningRate: number = 0.01;

    input: string[] = [];
    tokenizer = new natural.WordTokenizer();
    lastTokenIndex = 0;

    constructor(vocabularySize: number, dimensionality: number, learningRate: number = 0.001) {
        this.vocabularySize = vocabularySize;
        this.dimensionality = dimensionality;
        this.vocabulary = new Matrix(vocabularySize, dimensionality, () => 1000 * Math.random() - 500);
        this.vocabularyMGradient = new Matrix(vocabularySize, dimensionality, () => 0);
        this.vocabularyVGradient = new Matrix(vocabularySize, dimensionality, () => 0);
        this.learningRate = learningRate;
        this.finalFFB = FeedForwardBlock.Builder
            .setInput(dimensionality)
            .setOutput(vocabularySize, Softmax)
            .build();
    }

    assignToken(token: string) {
        if (this.lastTokenIndex >= this.vocabularySize) return;
        token = token.toLowerCase();
        this.dictionary.set(token, this.lastTokenIndex);
        this.reverseDictionary.set(this.lastTokenIndex, token);
        this.lastTokenIndex++;
    }

    getTokenIndex(token: string): number {
        token = token.toLowerCase();
        return this.dictionary.get(token) || 0;
    }

    getTokenEmmbedding(token: string): number[] {
        return this.vocabulary.data[this.getTokenIndex(token)];
    }

    getTokenAt(index: number) {
        return this.reverseDictionary.get(index);
    }

    addBlock(block: AttentionBlock | FeedForwardBlock | LayerNorm) {
        this.blocks.push(block);
    }

    forward(input: string[]): number[][] {
        let tokens = input.map(token => this.getTokenEmmbedding(token));
        this.blocks.forEach(block => {
            if (block instanceof AttentionBlock) {
                tokens = block.forward(new Matrix(tokens.length, this.dimensionality, tokens)).data;
            } else if (block instanceof LayerNorm) {
                tokens = block.forward(new Matrix(tokens.length, this.dimensionality, tokens)).data;
            } else {
                block.resetInputs();
                tokens = tokens.map(token => {
                    block.forward(token)
                    return [...block.output];
                });
            }
        });
        return tokens.map(token => {
            this.finalFFB.forward(token);
            return [...this.finalFFB.output];
        });
    }

    backpropagate(outputGradient: number[][]) {
        let lastGradient = this.finalFFB.backpropagateMultiple(outputGradient, this.learningRate);
        [...this.blocks].reverse().forEach(block => {
            if (block instanceof AttentionBlock) {
                lastGradient = block.backpropagate(lastGradient, this.learningRate).data;
            } else if (block instanceof LayerNorm) {
                lastGradient = block.backpropagate(lastGradient, this.learningRate).data;
            } else{
                lastGradient = block.backpropagateMultiple(outputGradient, this.learningRate);
            }
        });
        return lastGradient;
    }

    learnFromText(text: string) {

        const sequence = this.tokenizer.tokenize(text);
        sequence.forEach(token => {
            this.assignToken(token);
        });

        const input = sequence.slice(0, sequence.length - 1);
        const result = this.forward(input);
        let costSum = 0;
        const outputGradient = sequence.slice(1, sequence.length)
            .map((expectedToken, i) => {
                const a = new Array(this.dimensionality).fill(0);
                return a.map((_, j) => {
                    const goal = (j != this.getTokenIndex(expectedToken)) ? 0 : 1;
                    const cost = Math.pow(goal - result[i][j], 2);
                    costSum += cost;
                    return 2 * (result[i][j] - goal);
                })
            });
        costSum /= (sequence.length - 1);
        const finalGradient = this.backpropagate(outputGradient);

        input.forEach((token, i) => {
            const vI = this.getTokenIndex(token);
            this.vocabularyMGradient.data[vI].map((M, j) => {
                return M * this.BETA1 + finalGradient[i][j] * (1 - this.BETA1);
            })
            this.vocabularyVGradient.data[vI].map((V, j) => {
                return V * this.BETA1 + finalGradient[i][j] * finalGradient[i][j] * (1 - this.BETA1);
            })
            this.vocabulary.data[vI] = 
            this.vocabulary.data[vI].map((a, j) => {
                const correctedM = this.vocabularyMGradient.get(vI, j) / (1 - Math.pow(this.BETA1, this.learningT));
                const correctedV = this.vocabularyVGradient.get(vI, j) / (1 - Math.pow(this.BETA2, this.learningT));
                return a - (correctedM * this.learningRate) / (Math.sqrt(correctedV) + this.EPSILON);
            });
        });

        this.learningT++;

        return costSum;
    }

    predictNext(text: string) {

        const sequence = this.tokenizer.tokenize(text);
        sequence.forEach(token => {
            this.assignToken(token);
        });
;
        const result = this.forward(sequence);
        const lastDistribution = result[0];
        let maxIndex = 0, maxProb = 0;
        lastDistribution.forEach((prob, i) => {
            if (prob > maxProb) {
                maxIndex = i;
                maxProb = prob;
            }
        })
        return text + " " + this.getTokenAt(maxIndex);

    }

}