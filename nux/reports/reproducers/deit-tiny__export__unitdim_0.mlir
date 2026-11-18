func.func @repro(%arg0: tensor<1x192x14x14xf32>) -> tensor<1x192x196xf32> {
  %0 = tensor.collapse_shape %arg0 [[0], [1], [2, 3]] : tensor<1x192x14x14xf32> into tensor<1x192x196xf32>
  return %0 : tensor<1x192x196xf32>
}
