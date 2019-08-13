import TensorFlow

struct SimpleBatch: TensorGroup {
    let features: Tensor<Float>
    let labels: Tensor<Int32>
}

struct SimpleModel: Layer {
    var layer1 = Dense<Float>(inputSize: 1, outputSize: 1, activation: relu)
    var layer2 = Dense<Float>(inputSize: 1, outputSize: 3)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2)
    }
}

func accuracy1(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}

func runVerySimpleModel() {
    let batches = Dataset<SimpleBatch>(elements: SimpleBatch(
        features: [
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9],
        ],
        labels: [2, 1, 2, 1, 2, 1, 2, 1, 2]
    )).batched(1)

    let epochCount = 100
    var model = SimpleModel()
    let optimizer = SGD(for: model, learningRate: 0.01)

    for epoch in 1 ... epochCount {
        print("epoch \(epoch)")
        for batch in batches {
            var epochLoss: Float = 0
            var epochAccuracy: Float = 0

            print("From \(model)")
            let (loss, grad) = model.valueWithGradient { (model: SimpleModel) -> Tensor<Float> in
                let logits = model(batch.features)
                return softmaxCrossEntropy(logits: logits, labels: batch.labels)
            }
            print("loss: \(loss), With Grad: \(grad)")

            optimizer.update(&model.allDifferentiableVariables, along: grad)
            print("Optimized to: \(model.allDifferentiableVariables)")

            let logits = model(batch.features)
            print("logits \(logits), argmax: \(logits.argmax(squeezingAxis: 1))")
            epochAccuracy += accuracy1(predictions: logits.argmax(squeezingAxis: 1), truths: batch.labels)
            print("epochAccuracy \(epochAccuracy)")
            epochLoss += loss.scalarized()
            print("epochLoss \(epochLoss)")

            print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
        }
    }
}
