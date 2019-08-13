import TensorFlow


@differentiable
func sillyExp(_ x: Float) -> Float {
    return sin(x)
}

// @differentiating(sillyExp)
// func sillyDerivative(_ x: Float) -> (value: Float, pullback: (Float) -> Float) {
//     let y = sillyExp(x)
//     print("y", y)
//     return (value: y, pullback: {v in v * y})
// }

@differentiable
func silly2(_ x: Float, _ y: Float) -> Float {
	return x*x*x + y*y
}

let layer1 = Dense<Float>(inputSize: 4, outputSize: 10, activation: relu)
let layer2 = Dense<Float>(inputSize: 10, outputSize: 3)

@differentiable
func silly3(_ input: Tensor<Float>) -> Tensor<Float> {
	return input.sequenced(through: layer1, layer2)
}

func gradientRun() {
	print(sillyExp(Float.pi/2))
	print(gradient(of: sillyExp)(0.5))


	print(silly2(5, 1.5))
	print(gradient(of: silly2)(5, 1.5))
	print(gradient(of: silly3)(Tensor<Float>(zeros: [10, 4])))

    let layer1 = Dense<Float>(inputSize: 2, outputSize: 6, activation: relu)
    print("layer1 \(layer1), \(layer1.call(Tensor<Float>(zeros: [8, 2])))")
}
