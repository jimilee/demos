func.func @repro(%arg0: tensor<1x3x64x197xf32>) -> tensor<3x64x197xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<1x3x64x197xf32> into tensor<3x64x197xf32>
  return %0 : tensor<3x64x197xf32>
}
