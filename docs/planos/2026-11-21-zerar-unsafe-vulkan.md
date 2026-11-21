# Zerar o `unsafe` do backend Vulkan: opções e custo

Data: 2026-11-21. Situação: o usuário proibiu `unsafe` em Rust. O workspace já tem
`unsafe_code = "deny"` (`Cargo.toml`), e só `crates/llama-vulkan` o libera
(`[lints.rust] unsafe_code = "allow"`). Outro trabalho em andamento remove o `unsafe` que dá
para tirar sem trocar o `ash` e junta o resto em `crates/llama-vulkan/src/ffi.rs`, com wrappers
seguros. Este documento trata do que **sobra** depois disso: as chamadas ao `ash`, que são
todas `unsafe fn`.

Contagem atual de linhas com `unsafe` em `crates/llama-vulkan/src`: cerca de 205.
`resident_forward.rs` tem 67, `spike.rs` 38, `tensor.rs` 29, `matmul.rs` 23, `device.rs` 18,
`resident.rs` 13 e o resto tem menos de 10 cada. Fora do Vulkan ainda há `unsafe` em
`llama-model/src/ops.rs` (32, SIMD de CPU) e `spin_pool.rs` (9). Esses ficam fora do escopo
deste documento.

## Premissa que vale para todas as opções

Nenhuma opção elimina o `unsafe` do processo. O `ash`, o `wgpu-hal`, o `vulkano`, o `memmap2`
e o `libc` têm `unsafe` por dentro. O que se pode ter é "zero `unsafe` **nos crates do
projeto**", com `#![forbid(unsafe_code)]` em cada um. A pergunta real é qual dependência
carrega o `unsafe` e quanto desempenho e controle se perde com ela.

Duas coisas não são Vulkan e precisam de FFI em qualquer opção:

- **pstate** (`pstate.rs:57-60`): `ioctl(DRM_IOCTL_AMDGPU_CTX)` via `libc`, sem wrapper seguro
  em crate conhecido. Saídas possíveis: um crate próprio `*-sys` com esse `unsafe`, ou desistir
  do pstate pela ioctl. O sysfs `power_dpm_force_performance_level` é seguro mas exige root, e
  a memória `dpm-derruba-clock-no-layer-split` mediu 23,5 → 29,3 tok/s com o pstate.
- **mmap do modelo**: `memmap2::Mmap::map` é `unsafe fn`. Ele aparece em `llama-cli/src/runner.rs`,
  `llama-server/src/main.rs` e nos testes. A troca segura é `std::fs::read` para um `Vec`: custa
  uma cópia de 20–28 GB e o dobro do pico de RAM durante a carga (é uma estimativa). Com o
  problema de NUMA já registrado (`travamento-numa-node0-e-numactl`), isso é arriscado. A saída
  realista é manter o mmap no mesmo crate `*-sys`.

## O que os shaders usam hoje

Evidência em `crates/llama-vulkan/shaders/`: 32 `.comp` com 3721 linhas, compilados pelo
`shaderc` para SPIR-V 1.3 / Vulkan 1.1 (`build.rs`).

| Recurso | Uso | Onde |
|---|---|---|
| `GL_KHR_shader_subgroup_arithmetic` / `basic` | 17 / 16 shaders | `subgroupAdd` (24), `subgroupMax/Min`, `subgroupExclusiveAdd` (2) |
| `GL_KHR_shader_subgroup_clustered` | 1 shader | `moe_matvec.comp`: `subgroupClusteredAdd` com clusters de 8/16/32 (4 chamadas) |
| `GL_KHR_shader_subgroup_shuffle` | 2 shaders | `subgroupShuffleXor` (6) |
| `GL_EXT_integer_dot_product` | 9 shaders | `dotPacked4x8AccSatEXT` (74), `dotPacked4x8EXT` (2): o int8 dot4 de todos os matvec K-quant |
| f16 | `unpackHalf2x16` (45), `packHalf2x16` (3) | KV-cache f16. Não há `float16_t` nativo |
| `v_fma_mix_f32` | implícito | `attention_split16.comp:15`: o ACO funde `unpackHalf2x16` + `fma` |
| Specialization constants | 9 shaders | inclusive `local_size_x_id` (tamanho do workgroup) e tipo de quant (`mul_mm`, `moe_*`) |
| Shared memory | 13 shaders | tiles de `mul_mm*`, `atencao_bloco` (`uvec4 kl/vl[BC*UV_POS]`), `attention` (`s_acc[NWAVE*256]`) |
| `GL_EXT_control_flow_attributes` | 2 shaders | `[[unroll]]` |

Push constants: 4 a 28 bytes. 28 é o máximo, usado em `attention*` e `atencao_bloco`
(`resident_forward.rs:1584-1588`). Os layouts têm até 6 bindings de storage buffer.

## Custo de host por dispatch: o ponto que decide

O decode re-grava o command buffer **a cada token** (`iniciar`, `resident_forward.rs:~6100`:
`reset` → `record_token` → `submit`), e a gravação fica no caminho crítico. Cada op tem 4
comandos `vkCmd*` (`gravar_dispatch`, `:3469-3586`): bind pipeline, bind descriptor set, push e
dispatch. As barreiras só entram onde há hazard RAW/WAR/WAW (`marcar_barreiras`, +5,3% medido em
`docs/performance-tuning.md`).

**Dispatches por token no 35B-A3B (é uma estimativa, feita lendo `build_plan_n`, `plano_delta`
e `plano_moe`, não contada em execução).** Uma camada delta-net tem cerca de 19 dispatches: 2 da
norma, 9 do delta, 2 da norma do FFN e ~6 do MoE. Uma camada de atenção tem cerca de 22. São 40
camadas (30 delta + 10 atenção), o que dá **~800 dispatches/token**. A ~100 tok/s, o orçamento é
de 10 ms/token. Os dois shards correm em série no decode, então todos os dispatches estão no
caminho crítico.

O projeto já mediu que custos de host invisíveis no perfil de GPU valiam 18% do decode
(memória `custos-de-host-no-decode-gpu`: espera bloqueante do fence, map/unmap por token, readback
em memória WC).

**wgpu.** O dado público mais próximo é o arXiv 2604.02344 (2026). Ele mede o custo por dispatch
do wgpu-native em Vulkan, em sequência (sincronizando só no fim): **24,5 µs numa iGPU AMD e
35,8 µs numa RTX 5090**. O CUDA no mesmo trabalho fica em 7,4 µs. O número inclui o trabalho do
framework e provavelmente um encoder por op, então é um teto pessimista. Com tudo num único
`ComputePass` e bind groups pré-criados, estimo 2–5 µs por dispatch, já contando a validação e
o rastreamento de uso do `wgpu-core`. Essa faixa **não foi medida**.

| Custo por dispatch | × 800 | Fração de 10 ms |
|---|---:|---:|
| 2 µs (otimista, estimado) | 1,6 ms | +16% |
| 5 µs (estimado) | 4 ms | +40% |
| 24,5 µs (arXiv, iGPU AMD) | 19,6 ms | ~3× mais lento |

Há ainda dois riscos na mesma direção:

- O `wgpu-core` decide as barreiras sozinho, por buffer inteiro e por uso. Não há como reproduzir
  o agrupamento do `marcar_barreiras`. A perda máxima é o +5,3% medido, ou mais se o wgpu puser
  barreira entre dispatches que só leem/escrevem buffers RW distintos. Isso não foi verificado.
- O naga insere **bounds checks** em todo acesso a storage buffer. Desligá-los exige
  `Device::create_shader_module_trusted`, que é `unsafe fn` (`ShaderRuntimeChecks`). No matvec,
  que é limitado por banda, o custo deve ser pequeno, mas não foi medido.

Antes de qualquer decisão, vale medir o `t_rec` que `iniciar` já calcula (`:6120`), para saber
quanto a gravação com o `ash` custa hoje.

## Opções

### A. wgpu com shaders reescritos em WGSL

Ganha: API segura de ponta a ponta, desde que não se use `create_shader_module_trusted` nem
passthrough.

Perde ou arrisca:

- **Clustered**: o IR do naga 30.0.2 só tem `CollectiveOperation::{Reduce, InclusiveScan,
  ExclusiveScan}`, sem cluster. Dá para emular em `moe_matvec` com `subgroupShuffleXor` em log2
  passos (3–5 shuffles). É barato, mas exige reescrever e medir.
- **int8 dot4**: o naga tem `MathFunction::Dot4I8Packed` / `Dot4U8Packed` (WGSL `dot4I8Packed`),
  mas **não tem variante com acumulação saturada**. Os 74 `dotPacked4x8AccSatEXT` viram
  `dot4I8Packed(a,b) + acc`. A semântica só muda em overflow de i32, que não acontece em blocos
  de 32. Risco: o ACO precisa voltar a fundir em `v_dot4_i32_i8`, e o spv-out do naga precisa
  emitir `OpSDot`, não uma expansão escalar. Se não emitir, o matvec perde o dot4 (que vale 2× o
  FP16, segundo a memória `tetos-medidos-mi50`). **A verificar** com `naga x.wgsl x.spv` + ISA via
  `RADV_DEBUG=shaders`.
- **f16 / fma_mix**: `unpack2x16float` + `fma` existem em WGSL, e `enable f16` exige
  `Features::SHADER_F16`. Se o ACO ainda gera `v_fma_mix_f32` depende do SPIR-V que o naga emitir.
  **A verificar no ISA.**
- **Specialization constants** → `override` do WGSL, inclusive em `@workgroup_size`. Arrays de
  `var<workgroup>` dimensionados por `override` têm suporte no naga mais recente. A conferir na
  versão adotada.
- **Push constants**: `Features::IMMEDIATES` (antes `PUSH_CONSTANTS`) e
  `Limits::max_immediate_size` (Vulkan: 128–256 B). Os 28 B cabem.
- **Duas GPUs**: `enumerate_adapters` + um `Device` por adapter funciona. A escolha por PCI é
  possível, porque `AdapterInfo::device_pci_bus_id` existe no wgpu 30, então `device.rs:50-60`
  continua funcionando via sysfs. A cópia entre GPUs continua passando pelo host (a stream
  residual, ~20 KB/token), como hoje em `layer_split.rs`.
- **VRAM livre**: o wgpu não expõe o `VK_EXT_memory_budget`, que é a fonte de
  `free_device_memory` (`device.rs:141-161`) e da divisão de camadas. A alternativa segura é ler
  `mem_info_vram_used` no sysfs. Sem isso volta o caso em que a GPU do display empurra pesos
  para GTT (95 contra 714 GB/s).
- **Tipo de memória**: o wgpu escolhe o tipo, e não há como exigir `HOST_CACHED` como faz
  `tensor.rs:244-258`. Se o readback dos logits cair em WC, volta o custo medido de 4,2 ms/token
  contra 0,21 ms. **A verificar** que tipo o wgpu-hal escolhe para `MAP_READ`.
- **Mapeamento persistente**: proibido pela WebGPU. Um buffer precisa estar desmapeado para entrar
  num submit, então cada token faz `map_async` + `poll` + `unmap`, que é o padrão que o projeto
  removeu. Se isso vira ioctl ou só contabilidade interna depende do wgpu-hal. Não foi verificado.
- **Fence**: `device.poll` sem bloquear num laço substitui o `get_fence_status` sondado.
- **Prazo de 2 s do amdgpu**: dá para cortar o prefill em vários `queue.submit`, então não é
  bloqueio.
- **Tamanho de binding**: `max_storage_buffer_binding_size` vale 128 MiB por padrão. É preciso
  pedir o limite do adapter para os tensores grandes (embedding/output).
- Reescrita: 32 shaders (3721 linhas GLSL → WGSL) e toda a camada de host. Ver o custo no fim.

### B. wgpu carregando SPIR-V pelo naga (`ShaderSource::SpirV`, feature `spirv`)

Nessa rota os `.comp` e o `shaderc` ficam como estão. O SPIR-V passa pelo frontend `spv-in` do
naga e é re-emitido. Isso é **seguro**: só o `create_shader_module_passthrough` é `unsafe`.

Arrisca:

- O `spv-in` só aceita o que o IR do naga representa. Clustered e `OpSDotAccSat` não existem no
  IR, então `moe_matvec` e os 9 matvec seriam rejeitados até trocar o GLSL para shuffle e
  `dotPacked4x8EXT` + soma.
- O naga reescreve o módulo (com bounds checks) e pode não preservar o padrão que vira `fma_mix`.
- O resto do item A vale igual: custo por dispatch, memória e mapeamento.

É a rota de menor reescrita de shader dentro do wgpu. Exige ajustar uns 10 shaders, não 32.

### C. GLSL pelo frontend `glsl-in` do naga

O naga declara o `glsl-in` como "GLSL 440+ and Vulkan semantics only". Não há indício de suporte
às extensões `GL_KHR_shader_subgroup_*` nem a `GL_EXT_integer_dot_product` nesse frontend (não
verificado item a item). Na prática é pior que B, que lê o SPIR-V já gerado pelo `shaderc`.
Descartável.

### D. vulkano

- Shaders: o SPIR-V entra como hoje, com todas as extensões, specialization constants, push
  constants sem limite artificial e `HOST_CACHED`/mapeamento persistente via `MemoryTypeFilter`.
- **Não zera o `unsafe`.** O `dispatch` é `unsafe fn` na versão atual (vulkano não valida os
  acessos do shader), e criar `ShaderModule` a partir de SPIR-V também é `unsafe`. Reduz a
  contagem, mas não chega a zero. Isso já basta para descartar a opção, dada a regra.
- Custo de host: o vulkano também rastreia recursos por comando no `AutoCommandBufferBuilder`. A
  ordem de grandeza é desconhecida.

### E. Isolar o `ash` num crate `llama-vulkan-sys`

O `ffi.rs` vira um crate próprio com `unsafe_code = "allow"`, e `llama-vulkan` e todos os demais
ficam com `#![forbid(unsafe_code)]`. O pstate (ioctl) e o mmap vão para o mesmo crate, ou para um
`llama-sys` irmão.

- Não perde nada de shader, memória, duas GPUs nem custo de host. O código gerado é o mesmo de
  hoje.
- É **o mesmo modelo de confiança do wgpu**: o `unsafe` fica numa dependência. A diferença é que
  esta é do projeto e tem de ser auditada aqui. A superfície é pequena e conhecida (as chamadas
  `vkCmd*`, a alocação, o map e o submit).
- O risco é de disciplina: os wrappers precisam ser realmente seguros. Exemplos: um command
  buffer gravado não pode sobreviver aos buffers que referencia, e o push precisa ter o tamanho
  do range. Do contrário, o `unsafe` só muda de nome.
- O custo é o que o trabalho do `ffi.rs` já está fazendo, mais a separação em crate e o
  `forbid`.

### F. cubecl

Kernels escritos em Rust (`#[cube]`), sobre o runtime wgpu (WGSL ou SPIR-V) ou HIP. Herda o custo
por dispatch do wgpu e ainda soma o gerenciador de memória e o autotune do próprio cubecl. O
backend SPIR-V do cubecl chega ao driver por passthrough, com `unsafe` dentro do cubecl e não no
projeto. Não verifiquei se ele tem plane ops em cluster ou dot4 int8 com acumulação. O HIP não é
opção, porque o ROCm atual não suporta gfx906. Exige reescrever os 32 kernels numa DSL nova e não
oferece controle de barreira nem de tipo de memória. Não compensa.

### G. rust-gpu

Muda só a linguagem dos shaders (Rust → SPIR-V), com toolchain nightly fixada. O host ainda
precisa de `ash`, `vulkano` ou wgpu, então não resolve nada sozinho. Pior: as barreiras de
workgroup do `spirv-std` (`workgroup_memory_barrier_with_group_sync`) são `unsafe fn`, o que põe
`unsafe` dentro dos shaders. A confirmar na versão atual. Descartável para este objetivo.

## Recomendação

**Opção E.** Isolar o `ash`, o ioctl do pstate e o mmap num crate `llama-vulkan-sys` (ou
`llama-sys`), com `unsafe` só lá, e `forbid(unsafe_code)` em todo o resto. É a única opção que
cumpre a regra no código do projeto sem pôr em risco o decode de ~100 tok/s. O wgpu só cumpre a
regra de verdade se aceitar os bounds checks e o mapeamento por token, e mesmo assim deixa o
`unsafe` numa dependência, como a E.

Se o critério for "nenhum `unsafe` escrito por nós em lugar nenhum", a candidata é a **B (wgpu +
spv-in)**. Ela só deve ser adotada depois de um protótipo de 1 shader (`q4_k_matvec`) que meça:

- (i) µs por dispatch numa passada de 800 dispatches;
- (ii) a presença de `v_dot4_i32_i8` e `v_fma_mix_f32` no ISA;
- (iii) o tipo de memória do readback.

Custo estimado:

| Opção | Shaders | Host |
|---|---|---|
| E | 0 | mover `ffi.rs` para crate + pstate/mmap. Dias, não semanas (estimativa). |
| B | ~10 de 32 ajustados (clustered → shuffle; AccSat → dot + soma) | reescrever o host inteiro: `resident_forward.rs` (6524 linhas), `tensor.rs` (953), `matmul.rs` (1035), `device.rs` (445), `layer_split.rs` (492), `resident.rs` (452). ~10 mil das 12 293 linhas de `src/` mudam (estimativa) |
| A | 32 de 32 (3721 linhas GLSL → WGSL) | igual a B |
| D, F, G | — | não zeram o `unsafe` ou não resolvem o host |

Fontes externas: docs.rs `wgpu` 30.0.2 (`Features::IMMEDIATES`, `SUBGROUP`, `SHADER_F16`,
`AdapterInfo::device_pci_bus_id`, `create_shader_module_passthrough`/`_trusted` `unsafe`);
docs.rs `naga` 30.0.2 (`CollectiveOperation`, `MathFunction::Dot4I8Packed`); README do naga
(`glsl-in`); [arXiv 2604.02344](https://arxiv.org/html/2604.02344v1) (custo por dispatch do
WebGPU).
