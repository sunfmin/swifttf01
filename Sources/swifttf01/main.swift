import TensorFlow

import Python

//import Glibc

// gradientRun()

runVerySimpleModel()

exit(0)


struct Model: Differentiable {
    var w: Float
    var b: Float
    var b2: Float

    func run(to input: Float) -> Float {
        return w * input + b + b2
    }
}

let m = Model(w: 500, b: 10, b2: 20)
let (model1, input1) = m.gradient(at: -4.9) { model, input in
    model.run(to: input)
}
print(model1)
print(input1)

// exit(0)


print("\(softmax(Tensor<Float>([17.457628, 11.688464, -18.308298])))")
print("\(softmax(Tensor<Float>([1, 1, 1])))")
print("\(softmax(Tensor<Float>([1, 0, 0])))")
print("\(softmax(Tensor<Float>([2, 10, -100])))")
print("\(softmax(Tensor<Float>([1, -10])))")

extension String: Error {
}

// throw "quit"

let plt = Python.import("matplotlib.pyplot")

print(plt)

// import Foundation
// import FoundationNetworking

// func download(from sourceString: String, to destinationString: String) {
//     let source = URL(string: sourceString)!
//     let destination = URL(fileURLWithPath: destinationString)
//     let data = try! Data.init(contentsOf: source)
//     try! data.write(to: destination)
// }

// download(
//     from: "https://raw.githubusercontent.com/tensorflow/swift/master/docs/site/tutorials/TutorialDatasetCSVAPI.swift",
//     to: "TutorialDatasetCSVAPI.swift")

// let trainDataFilename = "iris_training.csv"
// download(from: "http://download.tensorflow.org/data/iris_training.csv", to: trainDataFilename)

let classNames = ["Iris setosa", "Iris versicolor", "Iris virginica"]


let featureNames = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
let labelName = "species"
let columnNames = featureNames + [labelName]

print("Features: \(featureNames)")
print("Label: \(labelName)")


let batchSize = 32

/// A batch of examples from the iris dataset.
struct IrisBatch {
    /// [batchSize, featureCount] tensor of features.
    let features: Tensor<Float>

    /// [batchSize] tensor of labels.
    let labels: Tensor<Int32>
}


let trainDataset: Dataset<IrisBatch> = Dataset(
        contentsOfCSVFile: "iris_training.csv", hasHeader: true,
        featureColumns: [0, 1, 2, 3], labelColumns: [4]
).batched(batchSize)

let firstTrainExamples = trainDataset.first!
let firstTrainFeatures = firstTrainExamples.features
let firstTrainLabels = firstTrainExamples.labels
print("First batch of features: \(firstTrainFeatures)")
print("First batch of labels: \(firstTrainLabels)")
print("firstTrainLabels.array.scalars \(firstTrainLabels.array.scalars)")

let firstTrainFeaturesTransposed = firstTrainFeatures.transposed()
print("firstTrainFeaturesTransposed \(firstTrainFeaturesTransposed)")


let petalLengths = firstTrainFeaturesTransposed[2].scalars
let sepalLengths = firstTrainFeaturesTransposed[0].scalars

print("petalLengths \(petalLengths)")
print("sepalLengths \(sepalLengths)")
print("firstTrainFeaturesTransposed[0] \(firstTrainFeaturesTransposed[0])")
print("firstTrainFeaturesTransposed[0].array \(firstTrainFeaturesTransposed[0].array)")

plt.scatter(petalLengths, sepalLengths, c: firstTrainLabels.array.scalars)
plt.xlabel("Petal length")
plt.ylabel("Sepal length")

//exit(0)
//
//plt.show()


let hiddenSize: Int = 10

struct IrisModel: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer4 = Dense<Float>(inputSize: hiddenSize, outputSize: 3)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3, layer4)
    }
}


var model = IrisModel()

let firstTrainPredictions = model(firstTrainFeatures)
print(softmax(firstTrainPredictions[0..<5]))

print("Prediction: \(firstTrainPredictions.argmax(squeezingAxis: 1))")
print("    Labels: \(firstTrainLabels)")

let untrainedLogits = model(firstTrainFeatures)
let untrainedLoss = softmaxCrossEntropy(logits: untrainedLogits, labels: firstTrainLabels)
print("Loss test: \(untrainedLoss)")

let optimizer = SGD(for: model, learningRate: 0.01)

let (loss, grads) = model.valueWithGradient { model -> Tensor<Float> in
    let logits = model(firstTrainFeatures)
    return softmaxCrossEntropy(logits: logits, labels: firstTrainLabels)
}
print("Current loss: \(loss)")

optimizer.update(&model.allDifferentiableVariables, along: grads)

let logitsAfterOneStep = model(firstTrainFeatures)
let lossAfterOneStep = softmaxCrossEntropy(logits: logitsAfterOneStep, labels: firstTrainLabels)
print("Next loss: \(lossAfterOneStep)")


let epochCount = 500
var trainAccuracyResults: [Float] = []
var trainLossResults: [Float] = []

func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}

for epoch in 1...epochCount {
    var epochLoss: Float = 0
    var epochAccuracy: Float = 0
    var batchCount: Int = 0
    for batch in trainDataset {
        let (loss, grad) = model.valueWithGradient { (model: IrisModel) -> Tensor<Float> in
            let logits = model(batch.features)
            return softmaxCrossEntropy(logits: logits, labels: batch.labels)
        }

        // print("Before optimize: \(model.allDifferentiableVariables)")
        optimizer.update(&model.allDifferentiableVariables, along: grad)
        // print("After optimize: \(model.allDifferentiableVariables)")

        let logits = model(batch.features)
        epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: batch.labels)
        epochLoss += loss.scalarized()
        batchCount += 1
    }
    epochAccuracy /= Float(batchCount)
    epochLoss /= Float(batchCount)
    trainAccuracyResults.append(epochAccuracy)
    trainLossResults.append(epochLoss)
    if epoch % 50 == 0 {
        print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
    }
}

plt.figure(figsize: [12, 8])

let accuracyAxes = plt.subplot(2, 1, 1)
accuracyAxes.set_ylabel("Accuracy")
accuracyAxes.plot(trainAccuracyResults)

let lossAxes = plt.subplot(2, 1, 2)
lossAxes.set_ylabel("Loss")
lossAxes.set_xlabel("Epoch")
lossAxes.plot(trainLossResults)

// plt.show()

let testDataset: Dataset<IrisBatch> = Dataset(
        contentsOfCSVFile: "iris_test.csv", hasHeader: true,
        featureColumns: [0, 1, 2, 3], labelColumns: [4]
).batched(batchSize)

for testBatch in testDataset {
    let logits = model(testBatch.features)
    let predictions = logits.argmax(squeezingAxis: 1)
    print("Test batch accuracy: \(accuracy(predictions: predictions, truths: testBatch.labels))")
}

let firstTestBatch = testDataset.first!
let firstTestBatchLogits = model(firstTestBatch.features)
let firstTestBatchPredictions = firstTestBatchLogits.argmax(squeezingAxis: 1)

print(firstTestBatchPredictions)
print(firstTestBatch.labels)


let unlabeledDataset: Tensor<Float> =
        [[5.1, 3.3, 1.7, 0.5],
         [5.9, 3.0, 4.2, 1.5],
         [6.9, 3.1, 5.4, 2.1]]

let unlabeledDatasetPredictions = model(unlabeledDataset)

for i in 0..<unlabeledDatasetPredictions.shape[0] {
    let logits = unlabeledDatasetPredictions[i]
    let classIdx = logits.argmax().scalar!
    print("Example \(i) prediction: \(classNames[Int(classIdx)]) (\(softmax(logits))) (\(logits))")
}

print("\(softmax(Tensor<Float>([17.457628, 11.688464, -18.308298]))) is [    0.9968874,  0.0031126477, 2.9221443e-16]")
