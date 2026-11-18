func.func @repro(%arg0: tensor<1x1x197x1xi1>) -> tensor<1x1x197xi1> {
  %0 = tensor.collapse_shape %arg0 [[0], [1], [2, 3]] : tensor<1x1x197x1xi1> into tensor<1x1x197xi1>
  return %0 : tensor<1x1x197xi1>
}
