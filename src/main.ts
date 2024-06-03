import NeuralNetwork from "./lib/NeuralNetwork";
import {TrainingMethod} from "./lib/builder/NeuralNetworkTrainingSession";
import NeuralLayer from "./lib/NeuralLayer";

function main() {

    const myNetwork = NeuralNetwork.Builder
        .setInput(5)
        .setOutput(1)
        .build();

    let session = myNetwork.startTrainingSession()
        .setMethod(TrainingMethod.STOCHASTIC)
        .setLearningRate(0.001);

    for(let i = 0; i < 100000; i++) {
        let input = [], sum = 0;
        for(let j = 0; j < 5; j++) {
            let n = Math.floor(Math.random() * 10 - 5);
            sum += n;
            input.push(n);
        }
        session.addData(input, [sum]);
    }
    session.end();

    myNetwork.activate([15, 15, -15, 2, -3]);
    console.log(myNetwork.toString());
    myNetwork.activate([1, 11, -10, 1, -3]);
    console.log(myNetwork.toString());
    myNetwork.activate([15, 0, 0, 0, -3]);
    console.log(myNetwork.toString());
    myNetwork.activate([1, 5, 2, 2, -3]);
    console.log(myNetwork.toString());
    myNetwork.activate([0, 0, 0, 2, -3]);
    console.log(myNetwork.toString());


}

main();