import NeuralNetwork from "./lib/NeuralNetwork";
import NeuralLayer from "./lib/NeuralLayer";

function main() {

    const myNetwork = NeuralNetwork.Builder
        .setInput(16)
        .setOutput(16)
        .addLayer(new NeuralLayer(8))
        .build();

    console.log(myNetwork.toString());

    myNetwork.activate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

    console.log(myNetwork.toString());

}

main();