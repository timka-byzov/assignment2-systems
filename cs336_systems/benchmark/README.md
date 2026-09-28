## timeit


Условия замера: `context_length = 512`, `batch_size = 10`.

| Модель | Forward, мс | Backward, мс | Optimizer, мс | Статус |
| --- | ---: | ---: | ---: | --- |
| small | 39.32 ± 6.65 | 69.86 ± 6.92 | 26.13 ± 3.28 | DONE |
| medium | 111.40 ± 6.69 | 212.17 ± 9.80 | 45.84 ± 12.54 | DONE |
| large | 232.98 ± 14.10 | 460.14 ± 18.36 | 70.21 ± 20.24 | DONE |
| xlarge | 649.25 ± 8.84 | 1295.11 ± 19.12 | 77.67 ± 5.78 | DONE |
| 10B | — | — | — | OOM на прогреве |
| large no warmup | 247.42 ± 145.90 | 457.47 ± 49.18 | 65.04 ± 18.41 | DONE |

видно что backward занмает ~ x2 forward. возможно это из за x2 matmuls (dW и dX)

## nsys_profile

`cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x128x16_1x1x1_3_tnn_align1_bias_f32_relu` - похоже это swiglu
вызывается в small модели {слои} x 2 раз


`cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bias_f32_relu` - вызывается только на backward
