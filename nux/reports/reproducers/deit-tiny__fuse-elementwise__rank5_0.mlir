func.func @repro(%arg0: tensor<1x1x197x1x197xf32>) -> tensor<1x1x197x197xf32> {
  %0 = tensor.collapse_shape %arg0 [[0], [1], [2, 3], [4]] : tensor<1x1x197x1x197xf32> into tensor<1x1x197x197xf32>
  return %0 : tensor<1x1x197x197xf32>
}
