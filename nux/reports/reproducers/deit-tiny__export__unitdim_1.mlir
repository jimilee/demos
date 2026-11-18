func.func @repro(%arg0: tensor<197xi64>) -> tensor<1x1x197x1xi64> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 1, 197, 1] : tensor<197xi64> into tensor<1x1x197x1xi64>
  return %0 : tensor<1x1x197x1xi64>
}
