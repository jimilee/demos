func.func @repro(%arg0: tensor<3x197x197xf32>) -> tensor<1x3x197x197xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2], [3]] output_shape [1, 3, 197, 197] : tensor<3x197x197xf32> into tensor<1x3x197x197xf32>
  return %0 : tensor<1x3x197x197xf32>
}
