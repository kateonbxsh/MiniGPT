import MultiLayerPerceptron from "@/lib/models/mlp/MultiLayerPerceptron";
import GPT from "./lib/models/gpt/GPT";
import AttentionBlock from "./lib/models/components/AttentionBlock";
import ReLU from "./lib/activation/ReLU";
import salutations from "./data/salutations";
import Softmax from "./lib/activation/Softmax";
import LeakyReLU from "./lib/activation/LeakyReLU";
import NeuralLayer from "./lib/models/components/NeuralLayer";
import MinMaxNorm from "./lib/activation/MinMaxNorm";
import LayerNorm from "./lib/models/components/LayerNorm";
import Sigmoid from "./lib/activation/Sigmoid";
import GELU from "./lib/activation/GELU";
import FeedForwardBlock from "./lib/models/mlp/FeedForwardBlock";

function main() {

    const tokenSize = 16;
    const gpt = new GPT(50, tokenSize, 0.001);

    gpt.addBlock(new AttentionBlock(tokenSize));
    gpt.addBlock(new LayerNorm(tokenSize));
    gpt.addBlock(FeedForwardBlock.Builder
        .setInput(tokenSize)
        .addLayer(new NeuralLayer(2 * tokenSize), GELU)
        .addLayer(new NeuralLayer(4 * tokenSize), Softmax)
        .setOutput(tokenSize)
        .build()
    );
    gpt.addBlock(new LayerNorm(tokenSize));
    gpt.addBlock(new AttentionBlock(tokenSize));
    gpt.addBlock(new LayerNorm(tokenSize));
    gpt.addBlock(FeedForwardBlock.Builder
        .setInput(tokenSize)
        .addLayer(new NeuralLayer(2 * tokenSize), GELU)
        .addLayer(new NeuralLayer(4 * tokenSize), Softmax)
        .setOutput(tokenSize)
        .build()
    );
    gpt.addBlock(new LayerNorm(tokenSize));
    gpt.addBlock(new AttentionBlock(tokenSize));
    gpt.addBlock(new LayerNorm(tokenSize));

    let lastAverage = 0, positiveVariations = 0;
    for(var i = 0; i >= 0; i++) {
        let averageCost = 0, totalNb = salutations.length;
        salutations.forEach(salutation => {
            let cost = gpt.learnFromText(salutation);
            if (isNaN(cost)) {
                console.log("DIVERGED!");
                process.exit(1);
            } else {
                averageCost+=cost/totalNb;
            }
        })
        console.log("Average cost", averageCost, 
            ", Epoch:", i+1, 
            ", Cost variation:", (averageCost - lastAverage))
        if ((averageCost - lastAverage) > 0) positiveVariations++;
        else positiveVariations = 0;
        lastAverage = averageCost;
        if (positiveVariations > 20) break;
    }
    
    let startText = "Hello";
    for(var i = 0; i < 20; i++) {
        console.log(startText);
        startText = gpt.predictNext(startText);
    }

}

main();