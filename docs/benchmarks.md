# Benchmarks

Todos os números abaixo são token generation (decode), não prompt processing — greedy ou
sampling padrão, medidos com `--timings` (mede da emissão do primeiro token em diante, exclui o
tempo de prefill do prompt). Hardware: 2× AMD MI50 (Radeon Pro VII, gfx906, 16 GB HBM2 cada).

## Conversa longa até 30k tokens (2026-09-25)

Mesma conversa nos dois servidores, decode medido com o contexto no cache (detalhes, prefill e
método em [`benchmark-conversa-longa-2026-09-25.md`](benchmark-conversa-longa-2026-09-25.md)):

| contexto | llama.rs | llama.rs + MTP | llama.cpp HIP | llama.cpp HIP + MTP |
|---:|---:|---:|---:|---:|
| 1.448 | **26,0** | **33,5** | 23,4 | 21,3 |
| 15.745 | **24,0** | **25,2** | 21,9 | 19,7 |
| 30.644 | **22,2** | 20,3 | 20,4 | 19,5 |

Desde `35ceb64` o `--mtp` só especula abaixo de 12k tokens; acima disso o decode é o da coluna
sem MTP.

Prefill: o llama.cpp é ~1,85× mais rápido (318 contra 171,9 tok/s com 7,5k, desde o bloco de
256). O snapshot na fronteira do turno (`1769c35`) levou o TTFT com 30k de 387 s para 31,5 s.

## Qwen3.8-27B (híbrido atenção + gated delta-net), 2× MI50 layer-split

| Config | tok/s |
|---|---:|
| llama-rs, Q5_K_M | 25.5 |
| llama-rs, Q4_K_M | **26.9** |
| llama.cpp, Vulkan, baseline (sem MTP) | 17.0 – 20.6 |
| llama.cpp, ROCm/HIP, baseline (sem MTP) | 20.4 – 23.4 |
| llama.cpp, ROCm/HIP, com MTP | 19.4 – 21.5 (abaixo do sem MTP) |

O intervalo do llama.cpp reflete profundidade de contexto — os dois extremos (prompt curto vs
~8K tokens de contexto acumulado) foram medidos separadamente. Os números do llama-rs foram
medidos com prompt curto, mesma classe de condição do extremo baixo do llama.cpp.

Essa é a arquitetura mais recente e mais complexa suportada: 64 camadas, 3 em cada 4 usam
**gated delta-net** (atenção linear com estado recorrente por cabeça, sem KV-cache) em vez de
atenção — ver [`qwen35-arquitetura.md`](qwen35-arquitetura.md). O layer-split não é só sobre
velocidade aqui: o modelo (15–20 GiB dependendo da quantização) não cabe numa única MI50 de 16 GB.

## Qwen2.5-32B Q5_K_M, 2× MI50 layer-split

| | tok/s | razão |
|---|---:|---:|
| **llama-rs** | 19.3 | **1.07×** |
| llama.cpp (ROCm) | 18.02 | — |

O caso que justifica dividir entre as duas GPUs: 22.2 GB não cabem numa placa só.

## Qwen2.5-14B Q8_0, 1× MI50

| | tok/s | razão |
|---|---:|---:|
| llama-rs | 28.0 | 0.69× |
| llama.cpp (Vulkan) | 40.59 | — |

Modelo que cabe inteiro numa GPU — aqui o llama.cpp ainda ganha.

## Qwen2.5-0.5B Q8_0, 1× MI50

| | tok/s | razão |
|---|---:|---:|
| llama-rs | 123 | 0.37× |
| llama.cpp (Vulkan) | 334 | — |

Modelo pequeno demais para o overhead por-dispatch do llama-rs se amortizar — o llama.cpp tem
kernels mais especializados para esse regime.

## Backend CPU (referência, sem GPU)

| Modelo | llama.cpp | llama-rs | razão |
|---|---:|---:|---:|
| stories260K (f32) | ~1000 tok/s | ~1045 tok/s | 1.04× |
| Qwen2.5-0.5B Q8_0 | ~12 tok/s | ~4.3 tok/s | 0.36× |

O foco do projeto é o backend Vulkan; o caminho CPU existe para desenvolvimento e para modelos
pequenos, não é otimizado com os mesmos kernels vetorizados que o llama.cpp usa em AVX2.

## O que já foi tentado e descartado

- **Tensor-parallel / row-split entre as 2 GPUs** — sem P2P de VRAM neste hardware, o custo de
  sincronização por camada (96 all-reduces/token num modelo de 14B) supera a banda que
  economizaria. Ver [`estrategia-inferencia-mi50.md`](estrategia-inferencia-mi50.md).
- **MTP no llama.cpp** — neste hardware/modelo o MTP do llama.cpp perde em todas as
  profundidades. O do llama.rs (`--mtp`) rende +31% com 2k e +20% com 8k (Qwen3.8-27B Q4_K_M,
  2026-11-20: 34,4/29,3 contra 26,3/24,4 tok/s) e desliga acima de 12k. Ver
  [`mtp-e-k80.md`](mtp-e-k80.md).
