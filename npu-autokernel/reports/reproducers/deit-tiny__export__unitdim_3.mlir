func.func @repro(%arg0: tensor<1x197x192xf32>) -> tensor<197x192xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x197x192xf32> into tensor<197x192xf32>
  return %0 : tensor<197x192xf32>
}
