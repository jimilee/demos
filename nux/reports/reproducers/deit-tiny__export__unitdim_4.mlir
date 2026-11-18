func.func @repro(%arg0: tensor<197x192xf32>) -> tensor<1x197x3x64xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [1, 197, 3, 64] : tensor<197x192xf32> into tensor<1x197x3x64xf32>
  return %0 : tensor<1x197x3x64xf32>
}
