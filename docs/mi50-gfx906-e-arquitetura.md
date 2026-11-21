# MI50 / gfx906 e a arquitetura do motor

**Data:** 2026-11-20 · **Fontes:** este repositório (`README.md`, `docs/hardware.md`,
`docs/benchmarks.md`, `docs/decode-por-configuracao.md`, `docs/prefill-em-batch.md`,
`docs/estrategia-inferencia-mi50.md`, `docs/planos/`) e o ai-memory
(`hardware/mi50-gfx906-e-motor.md`). Revisado em 2026-11-20 contra o código e as decisões de
2026-09-26, com medição nova depois do undervolt da card1.

## Identificação

| | |
|---|---|
| Repositório | `/home/murilo/llama.rs-MI50` |
| Remoto | `origin` = `git@github.com:MuriloCarvalhoo/llama.rs-MI50.git` (o remoto `llama-rs` ainda aponta para o espelho) |
| Commits | 292 em 2026-10-01 (projeto nascido em junho de 2026) |
| Toolchain | Rust 1.98.0 |
| Licença | MIT |

Runtime de inferência escrito **do zero** em Rust, sem ggml/llama.cpp em tempo de execução,
com backend Vulkan próprio e shaders SPIR-V feitos à mão para gfx906.

Em 2026-10-01 o nome `MuriloCarvalhoo/llama.rs` passou a hospedar o **porte espelho** do
llama.cpp, e o motor próprio migrou para `llama.rs-MI50`. São dois projetos: o espelho aposta
em fidelidade bit-exata ao C++, este aposta em controle total sobre shaders e agendamento de
memória no hardware.

## Placas alvo

| GPU | Arquitetura | VRAM | API |
|---|---|---|---|
| AMD MI50 / Radeon Pro VII | GCN 5.1, gfx906 | 16 GB HBM2 | Vulkan 1.1+ |
| AMD MI60 | GCN 5.1, gfx906 | 32 GB HBM2 | Vulkan 1.1+ |

Wavefront wave64, PCIe, **sem P2P/NVLink/Infinity Fabric entre placas**. O backend enumera só
devices AMD (vendor ID): é escopo, não limitação do Vulkan.

Medido nas duas placas desta máquina (`scripts/teto-mi50.sh`, 2026-09-25), contra o nominal:
leitura HBM2 **836 GB/s** (nominal 1024), FP32 12,75 TFLOPS (13,4), FP16 empacotado 25,2 (26,8),
`v_dot2_f32_f16` 25,4, INT8 dot4 51,0 TOPS (53,6), FP64 6,1 (6,7). O clock máximo do núcleo
nestas placas é **1700 MHz** (`pp_dpm_sclk`). 60 CU.

## O que o gfx906 decide nos kernels

- **wave64 nativo.** Os shaders assumem wave64. GPUs wave32 (maioria NVIDIA, AMD RDNA) não são
  suportadas. Tesla K80 foi descartada por isso.
- **Dot empacotado int8.** `dotPacked4x8AccSatEXT` compila para `v_dot4_i32_i8` no RADV (visto
  em desassemblagem). É o que dá ~51 TOPS de INT8 no prefill contra 836 GB/s de banda.
- **Sem MFMA.** Nenhum `v_mfma` aparece no disassembly; gfx908/MI100 têm, gfx906 não. Não há
  matrix cores.
- **Banda é o teto do decode batch-1.** Em pstate `min_sclk` o matvec Q4_K lê 380–430 GB/s; em
  `standard` (1316 MHz), 564–621 GB/s. Acima do `standard` o clock não rende: em 2026-11-20 o
  `peak` (1700 MHz) deu o mesmo decode que o `standard` (35,1 contra 34,4 tok/s a 2k, com MTP).
- **Atenção longa não é limitada por conta** (decisão 12, 2026-09-26): cortar 30% das operações
  do laço deu o mesmo tempo. Ela acompanha o clock do núcleo; a suspeita é a L2, que no Vega roda
  nesse clock e serve as releituras do grupo GQA. Por isso o `V_DOT2_F32_F16` na atenção foi
  descartado. O KV f16 empacotado cortou a VRAM do cache pela metade e não mudou o tempo.

## Arquitetura do motor

Workspace de 9 crates de runtime mais um crate `oracle`:

| Crate | Papel |
|---|---|
| `gguf` | Parser GGUF v3, zero-copy sobre slice |
| `llama-tokenizer` | SPM (Llama) e BPE (Qwen2/GPT-2), bit-exato contra llama.cpp |
| `llama-model` | Forward pass: attention, RMSNorm, RoPE, SwiGLU, gated delta-net, MTP, sessão |
| `ggml-cpu` | Dequantização e ops GGML de baixo nível no CPU |
| `llama-vulkan` | Backend Vulkan: shaders, decode residente, layer-split, dual_gpu, pstate |
| `llama-sampling` | Estratégias de amostragem |
| `llama-chat` | Chat template Qwen3.5/3.8 (ChatML + tools XML), provado contra Jinja real |
| `llama-cli` | CLI de geração |
| `llama-server` | Servidor HTTP compatível com a API OpenAI |
| `oracle` | Comparação de comportamento contra o C++ |

Vulkan: **24 shaders `.comp`, 2 625 linhas**. Módulos principais: `resident_forward.rs`
(~244 KB), `tensor.rs`, `matmul.rs`, `spike.rs`, `alloc.rs`, `device.rs`, `layer_split.rs`,
`dual_gpu.rs`, `pstate.rs`.

### Layer-split

As camadas dividem-se entre as duas GPUs. Existe por **capacidade** (Qwen3.8-27B de 15–20 GiB
não cabe numa MI50 de 16 GB), não por velocidade. A fronteira deriva da VRAM livre e varia entre
execuções: já mudou 139/149 para 134/154 e move o resultado ~2%. Em 2026-11-20 a automática deu
30/34 no Qwen3.8-27B. Fixar com `LLAMA_RS_SPLIT=N` antes de comparar builds.

Tensor-parallel/row-split foi **descartado**. Os 1,4–1,7× de TP=2 sobre PCIe vêm da literatura
(ver `estrategia-inferencia-mi50.md`), não de medição aqui. O custo é latência × 96
sincronizações por token.

### MTP

Ligado por `--mtp`, desligado por padrão. Duas propostas encadeadas, verify de 3 tokens. Corte em
`LLAMA_RS_MTP_ATE`, padrão 12 000 tokens (`LLAMA_RS_MTP_ATE=0` liga em toda posição): acima
disso o decode é simples. Medição controlada da decisão 11 (MTP em todo passo contra sem MTP):
+3% com 1k, empate com 8k, −12 a −21% com 16k, −13% com 29k. O ganho no curto depende do texto:
+29% numa conversa de 20 turnos com 1,4k; em prompts de código (2026-11-20), +31% com 2k e +20%
com 8k. Uma escolha adaptativa por passo foi tentada e descartada (estimativa enviesada contra o
MTP).

### Servidor

Rotas `POST /v1/chat/completions` (com e sem `stream`), `GET /v1/models`, `GET /health`. Uma
geração por vez, fila até 4, 503 com `Retry-After` além disso. Timeouts de socket: leitura 30 s
(408), escrita 60 s (cancela a geração). Parâmetro inválido responde 400. `model` diferente do
`--nome` não é erro: o servidor responde com o modelo que serve e registra no log.

O tool call deste modelo é **XML**, não JSON; a resposta começa dentro do `reasoning` (`<think>`),
que não é conteúdo.

### Térmica e NUMA

Escada do `pstate.rs`: 102/95 °C para a operação e 108 °C no monitor, contra o teto de 110 °C da
MI50. Em 2026-11-20 a card1 (PCI 05:00) recebeu undervolt: nos prompts de 2k a 30k ela chegou a
no máximo 89 °C em `standard` e 96 °C em `peak`, sem acionar a escada. Antes passava de 100 °C em
prefill sustentado. `numactl --interleave=all` não é opcional: sem ele o nó 0 estoura e a máquina
trava.

## Benchmarks medidos

Condições completas em [`benchmarks.md`](benchmarks.md) e
[`benchmark-conversa-longa-2026-09-25.md`](benchmark-conversa-longa-2026-09-25.md).

Qwen3.8-27B Q4_K_M, 2× MI50, layer-split, contexto 32 768, decode em tok/s, conversa crescente
de 2026-09-25 (a coluna + MTP é de antes do corte em 12k):

| Profundidade | llama.rs | llama.rs + MTP | llama.cpp HIP | llama.cpp HIP + MTP |
|---:|---:|---:|---:|---:|
| 1 448 | 26,0 | 33,5 | 23,4 | 21,3 |
| 7 656 | 25,1 | 27,8 | 22,8 | 21,5 |
| 15 745 | 24,0 | 25,2 | 21,9 | 19,7 |
| 22 378 | 23,1 | 23,4 | 21,1 | 19,4 |
| 30 644 | 22,2 | 20,3 | 20,4 | 19,5 |

Prompts de código, frio, 512 tokens gerados, 1 execução cada (2026-11-20, depois do undervolt):

| Contexto | Prefill | Decode sem MTP | Decode com MTP |
|---:|---:|---:|---:|
| 2 036 | 170 | 26,3 | 34,4 |
| 8 012 | 153 | 24,4 | 29,3 |
| 15 945 | 126 | 22,5 | 22,5 (acima de 12k) |
| 29 867 | 93 | — | 19,6 (acima de 12k) |

O llama.cpp HIP faz 318 tok/s de prefill no frio de 7,6k. O llama.rs fazia 115 em 2026-09-25 e
171,9 com 7,5k depois do bloco de 256 tokens (item 1.2).

Outros: Qwen2.5-32B Q5_K_M em layer-split, 19,3 contra 18,0 do llama.cpp (ROCm), razão 1,07×.
Qwen2.5-14B Q8_0 numa MI50 só, 28,0 contra **40,6** do llama.cpp (Vulkan) — quando o modelo cabe
numa placa, o backend genérico ganha.

Numa conversa de vários turnos o motor refazia o prefill do contexto inteiro a cada turno: com
30k tokens, TTFT de 387 s. O snapshot depois do último token especial (`1769c35`) baixou para
31,5 s, e agora só o turno novo passa pelo prefill.

## Limitações conhecidas

- Prefill ~1,85× mais lento que o llama.cpp HIP (171,9 contra 318 tok/s com 7,5k).
- Um snapshot só por conversa: divergência **antes** dele refaz o prompt inteiro.
- Perde quando o modelo cabe numa placa só.
- MoE só com experts Q3_K ou IQ3_XXS (os tipos do Qwen3.8-35B-A3B APEX); prefill do MoE 2,4–3,3× atrás do llama.cpp.
- Só AMD/wave64; sem NVIDIA, sem RDNA.
- Sem visão, `logprobs`, `n > 1`, penalidades de repetição.
- 48 das 65 camadas do qwen35 são delta-net, com estado recorrente: o KV de atenção volta atrás,
  o estado recorrente não.

## Decisões já tomadas — não refazer

1. Tensor-parallel/row-split não compensa neste hardware.
2. Decode batch-1 é limitado por banda de memória, inclusive nos K-quants — não por ALU.
3. K-quants trocam bytes por ALU exatamente onde a placa é fraca: projeções por bytes são limite
   superior otimista, não previsão.
4. Fusão de norma foi medida e rejeitada (`65e9614`).
5. Não forçar a divisão 31/33: a automática ganhou.
6. Testes sempre deixam ≥ 2 GB livres na GPU do monitor (regra do usuário, em `_rules`).
7. `RADV_DEBUG=shaders` altera o throughput ponta a ponta; tok/s medido assim não serve para
   comparar. (Fonte: ai-memory; não há medição registrada no repo.)

## Lacunas e contradições abertas

- As decisões 2 e 3 puxam em sentidos opostos: o Q4_K acompanha o clock até o `standard`, mas
  acima dele (`peak`) o decode não muda. Falta separar o que é banda e o que é ALU por kernel.
- Matvec do verify roda a 281 GB/s contra 450 do decode, e é 33% do verify.
- Delta-net e convolução em batch (o laço sobre os tokens dentro do kernel) continuam por fazer; é
  o que destrava o bloco de prefill grande.
- Largura de bloco ótima do prefill segue em aberto (padrão 256/32 com GEMM e passo de K = 64).
