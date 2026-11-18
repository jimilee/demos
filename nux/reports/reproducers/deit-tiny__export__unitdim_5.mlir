func.func @repro(%arg0: tensor<1x3x197x64xf32>) -> tensor<3x197x64xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<1x3x197x64xf32> into tensor<3x197x64xf32>
  return %0 : tensor<3x197x64xf32>
}
