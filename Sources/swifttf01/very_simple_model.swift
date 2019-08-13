import TensorFlow

struct SimpleBatch: TensorGroup {
    let features: Tensor<Float>
    let labels: Tensor<Int32>
}

struct SimpleModel: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: 10, activation: relu)
    var layer2 = Dense<Float>(inputSize: 10, outputSize: 10, activation: relu)
    var layer3 = Dense<Float>(inputSize: 10, outputSize: 3)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

func accuracy1(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}

func runVerySimpleModel() {
    // let batches = Dataset<SimpleBatch>(elements: SimpleBatch(
    //     features: [
    //         [1],
    //         [2],
    //         [3],
    //         [4],
    //         [5],
    //         [6],
    //         [7],
    //         [8],
    //         [9],
    //     ],
    //     labels: [0, 1, 0, 1, 0, 1, 0, 1, 0]
    // )).batched(2)

    let batches = Dataset<SimpleBatch>(elements: SimpleBatch(
        features: [[6.4, 2.8, 5.6, 2.2],
                   [5.0, 2.3, 3.3, 1.0],
                   [4.9, 2.5, 4.5, 1.7],
                   [4.9, 3.1, 1.5, 0.1],
                   [5.7, 3.8, 1.7, 0.3],
                   [4.4, 3.2, 1.3, 0.2],
                   [5.4, 3.4, 1.5, 0.4],
                   [6.9, 3.1, 5.1, 2.3],
                   [6.7, 3.1, 4.4, 1.4],
                   [5.1, 3.7, 1.5, 0.4],
                   [5.2, 2.7, 3.9, 1.4],
                   [6.9, 3.1, 4.9, 1.5],
                   [5.8, 4.0, 1.2, 0.2],
                   [5.4, 3.9, 1.7, 0.4],
                   [7.7, 3.8, 6.7, 2.2],
                   [6.3, 3.3, 4.7, 1.6],
                   [6.8, 3.2, 5.9, 2.3],
                   [7.6, 3.0, 6.6, 2.1],
                   [6.4, 3.2, 5.3, 2.3],
                   [5.7, 4.4, 1.5, 0.4],
                   [6.7, 3.3, 5.7, 2.1],
                   [6.4, 2.8, 5.6, 2.1],
                   [5.4, 3.9, 1.3, 0.4],
                   [6.1, 2.6, 5.6, 1.4],
                   [7.2, 3.0, 5.8, 1.6],
                   [5.2, 3.5, 1.5, 0.2],
                   [5.8, 2.6, 4.0, 1.2],
                   [5.9, 3.0, 5.1, 1.8],
                   [5.4, 3.0, 4.5, 1.5],
                   [6.7, 3.0, 5.0, 1.7],
                   [6.3, 2.3, 4.4, 1.3],
                   [5.1, 2.5, 3.0, 1.1]],
        labels: [2, 1, 2, 0, 0, 0, 0, 2, 1, 0, 1, 1, 0, 0, 2, 1, 2, 2, 2, 0, 2, 2, 0, 2, 2, 0, 1, 2, 1, 1, 1, 1]
    )).batched(2)

    let epochCount = 20
    var model = SimpleModel()
    let optimizer = SGD(for: model, learningRate: 0.01)

    for epoch in 1 ... epochCount {
        // print("epoch \(epoch)")
        var epochLoss: Float = 0
        var epochAccuracy: Float = 0
        var batchCount: Int = 0

        for batch in batches {
            // print("From \(model)")
            let (loss, grad) = model.valueWithGradient { (model: SimpleModel) -> Tensor<Float> in
                let logits = model(batch.features)
                return softmaxCrossEntropy(logits: logits, labels: batch.labels)
            }
            // print("loss: \(loss), With Grad: \(grad)")

            optimizer.update(&model.allDifferentiableVariables, along: grad)
            // print("Optimized to: \(model.allDifferentiableVariables)")

            let logits = model(batch.features)
            // print("logits \(logits), argmax: \(logits.argmax(squeezingAxis: 1))")
            epochAccuracy += accuracy1(predictions: logits.argmax(squeezingAxis: 1), truths: batch.labels)
            // print("epochAccuracy \(epochAccuracy)")
            epochLoss += loss.scalarized()
            // print("epochLoss \(epochLoss)")
            batchCount += 1
        }
        epochAccuracy /= Float(batchCount)
        epochLoss /= Float(batchCount)

        print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
    }
}
