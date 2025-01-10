# MiniGPT

MiniGPT is a modular GPT-like AI library written in TypeScript. The library lets you define the components or "blocks" that make up your own GPT-style model. You can mix and match different building blocks such as attention, feedforward layers, and more, enabling full customization for your specific needs. 

I wrote this library **from scratch**, implementing all the intrinsic calculations required for building and training the model, including gradient computation, forward propagation, and backpropagation. This hands-on approach allowed me to gain a deeper understanding of how the underlying mechanics of a GPT model function.


## Features

- **Modular Design**: Customize the architecture by defining the blocks of your GPT model.
- **Layer Components**: Includes a variety of core building blocks such as Multilayer Perceptron (MLP), LayerNorm, FeedForward (derived from MLP), and AttentionBlock.
- **Extensible**: Easily add more custom blocks or modify existing ones to suit your project.
- **TypeScript-based**: Fully written in TypeScript, providing strong typing and smooth integration with modern applications.

## Usage

### Basic Example: Building a Model

```typescript
const tokenSize = 4;
const gpt = new GPT(50, tokenSize, 0.0028);

gpt.addBlock(FeedForwardBlock.Builder
    .setInput(tokenSize)
    .addLayer(new NeuralLayer(2 * tokenSize), LeakyReLU())
    .setOutput(tokenSize)
    .build()
);
gpt.addBlock(new AttentionBlock(tokenSize));
gpt.addBlock(new LayerNorm(tokenSize));

gpt.learnFromText("hello world!");
console.log(gpt.predictNext("hi"));
```

## Project Background

This project was created as a way to explore and discover deep learning and machine learning concepts. During the development, I delved into optimization algorithms, neural network architectures, and the intricacies of passing gradients through multiple blocks. To improve training stability and performance, I implemented **LayerNorm** and **residual connections**. The project also uses the **Adam optimizer**, which is known for its efficiency in optimizing neural networks with sparse gradients.

This modular approach allows for easy experimentation and provides insights into how different components of a GPT-like model interact with each other.


## Core Blocks

- **MLP**: The Multilayer Perceptron class that can be used to build basic dense layers.
- **LayerNorm**: A class to apply Layer Normalization.
- **FeedForward**: A feedforward neural network block, a child of the MLP class, with the possibility of passing multiple inputs at the same time.
- **AttentionBlock**: A block implementing attention mechanisms (e.g., multi-head attention).

## Contributing

Feel free to fork the project, open issues, and submit pull requests. Contributions are always welcome!

## License

MiniGPT is open-source software released under the MIT License.
