# MTP no llama-rs: o que está pronto e o que falta

Multi-token prediction usando a cabeça que o próprio Qwen3.8-27B traz
(`nextn_predict_layers = 1`). O objetivo é gerar 2 tokens por passo de forward, já que o
passo é dominado por **ler os pesos** e ler os mesmos 16,38 GB serve para 1 ou 2 tokens.

`tok/s = base × (1 + taxa de aceitação)`. A aceitação foi **medida: 60,9 %** (fator 1,61),
o que torna o alvo de 50 tok/s alcançável — ver `docs/decode-por-configuracao.md`.

**Estado:** fases A–E implementadas; `--mtp` liga o caminho no `llama-cli` e no
`llama-server` (desligado por padrão). A cabeça propõe 2 tokens encadeados e o verify é de 3
(`VERIFY_TOK = 3`). A especulação só roda com o KV-cache abaixo de 12k (`MTP_ATE`;
`LLAMA_RS_MTP_ATE=N` muda, `0` = sempre): acima disso o verify perde para o decode simples.
Uma escolha adaptativa por passo foi tentada e descartada — media o MTP com viés contra
ele (decisão 11 de `planos/2026-09-26-decisoes-autonomas.md`). Medido em 2026-11-20 no
Qwen3.8-27B Q4_K_M: 34,4 / 29,3 tok/s com MTP a 2k / 8k de contexto, contra 26,3 / 24,4
sem; 2,20 tokens por passo em geração real greedy.

## Fluxo

```text
passo normal   h_t     = forward(t)                    → logits → amostra t+1
cabeça MTP     h_mtp   = eh_proj([enorm(emb(t+1)) ; hnorm(h_t)])   // 2*n_embd → n_embd
               h'      = camada_mtp(h_mtp)
               logits' = output(shared_head_norm(h'))  → propõe t+2
verificação    forward_batch([t+1, t+2])               → logits de t+2 e de t+3
               argmax(logits[t+1]) == t+2 ?  aceita os dois, segue de t+3
                                             senão, aceita só t+1
```

O ganho vem da verificação: `forward_batch` de 2 tokens lê os pesos **uma vez**, então
custa quase o mesmo que 1 token. É por isso que o batch é pré-requisito — sem ele o MTP só
adiciona trabalho. O fluxo acima é o de uma proposta; em produção a cabeça roda de novo
sobre a própria previsão e o verify cobre `[t+1, t+2, t+3]`.

## Fase A — carregar o bloco (PRONTA)

`MtpRaw` e `MtpAux` em `crates/llama-model/src/gpu.rs`, com
`crates/llama-model/tests/mtp_load.rs` validando as formas contra o GGUF.

**A pegadinha que o teste trava:** o bloco MTP é o `blk.64`, e `eh_linear(64)` responde
`true` porque `(64 + 1) % 4 != 0`. Seguir essa regra faria o carregador procurar
`ssm_out.weight`. O GGUF traz `attn_q/k/v/output` — **o bloco MTP é de atenção**, mesmo
numa posição que seria de atenção linear. Consequência boa: ele não tem estado recorrente
próprio.

Custo do bloco, medido: **289 MB → 0,40 ms** a 717 GB/s, ~0,9% de um passo de 44,8 ms.
Barato o bastante para não atrapalhar mesmo com aceitação baixa.

| tensor | forma | tipo |
|---|---|---|
| `nextn.eh_proj.weight` | 10240 → 5120 | Q8_0 |
| `nextn.enorm` / `hnorm` / `shared_head_norm` | 5120 | F32 |
| `attn_q` | 5120 → 12288 | Q4_K |
| `attn_k` / `attn_v` | 5120 → 1024 | Q4_K / Q6_K |
| `attn_output` | 6144 → 5120 | Q4_K |
| `ffn_gate` / `ffn_up` | 5120 → 17408 | Q4_K |
| `ffn_down` | 17408 → 5120 | Q6_K |

## Fase B — executar a cabeça (PRONTA: CPU como oráculo, GPU em produção)

`MtpHead::propor` em `crates/llama-model/src/mtp.rs` executa o bloco e devolve a proposta.
Roda **na CPU** de propósito: o que se queria era a taxa de aceitação, que não depende da
velocidade, e isso evitou montar um plano Vulkan antes de saber se o ganho existia.

Medido pelo `crates/llama-vulkan/tests/mtp_aceitacao.rs`, greedy, 23 propostas verificadas
contra o token que o modelo de fato produziu: **14 acertos, 60,9 %**.

Os 60,9 % também servem de prova de que a cabeça está correta — ela espelha a camada de
atenção do qwen35 incluindo dois detalhes fáceis de errar (queries com stride
`2 × head_dim`, porque o portão mora ao lado de cada uma; e o portão entrando como
`sigmoid` sobre a saída da atenção). Qualquer divergência derrubaria a aceitação para perto
de zero.

O matvec desquantiza **uma linha por vez** em vez de materializar o tensor: o
`output.weight` tem 248 320 linhas e custaria 5 GB de RAM só para o vocabulário.

A cabeça de CPU continua sendo o **oráculo**: é contra ela que a versão de GPU é validada
(`mtp_verify.rs::cabeca_mtp_na_gpu_bate_com_a_referencia_de_cpu`), e é ela que roda os
experimentos que não precisam de velocidade — como a aceitação encadeada da tarefa 6.

**Na GPU** (`MtpBufs` + `build_plan_mtp` em `resident_forward.rs`) a cabeça é um plano
próprio, montado só quando o backend é construído com MTP, e no shard que tem a norma final:

| passo | op |
|---|---|
| `enorm(emb)` e `hnorm(h)` | `NormFused` + `NormP2`, escrevendo nos offsets 0 e `n_embd` de `b_eh` — a concatenação é só binding com offset |
| quantizar os `2 * n_embd` | `QuantizeX` (cada `norm_p2` quantizou só a sua metade) |
| `eh_proj` | matvec Q8_0 |
| camada de decoder | o plano de uma camada de atenção do qwen35, com KV-cache próprio |
| `shared_head_norm` | norma |
| `output` | matvec Q6_K, a projeção **compartilhada** com o modelo |

Quase todas as ativações são emprestadas do plano principal: a cabeça roda **entre** dois
passos, com a GPU ociosa e os logits do passo anterior já lidos. O que precisa ser próprio é
o que sobrevive ao passo (`kcache`/`vcache`, 8 KB por posição) ou o que seria destruído antes
de ser lido (`b_h`, `b_eh`, `b_x`). A linha de embedding do token vem do host (20 KB), porque
a tabela só existe no primeiro shard e a cabeça mora no último.

**Risco conhecido:** mudar o número de bindings de um shader sem mudar o
`ComputePipeline::with(..., n_bindings, ...)` não dá erro nenhum — o modelo só passa a
gerar o último token do vocabulário, porque a saída vai para lugar nenhum. Já aconteceu
neste projeto. Validar cada op nova contra referência de CPU antes de encadear.

## Fase C — decode em batch e o plano de verify (PRONTA)

O plano generaliza para `n_tok` em vez de ganhar um irmão duplicado: o token do bloco vem
sempre de `gl_WorkGroupID.y`, então com `n_tok=1` cada shader colapsa no caminho de decode
de antes — o que os 28 testes de integração existentes confirmam a cada mudança.

**Shaders prontos e testados:**

| shader | como | commit |
|---|---|---|
| `q4_k/q5_k/q6_k_matvec` | spec constant `COLS`, N colunas por leitura de peso | `d93bb68`, `5a4d2fc` |
| `attention` | `gl_WorkGroupID.y` é o token; máscara causal por laço curto | `bacc81f` |
| `norm_fused` / `norm_p2` | parciais e `xq`/`xd` separados por token | `f9434ca` |
| `rope` | `pos - (n_tok - 1) + t` por token | `f9434ca` |
| `quantize_x`, `gate_quant`, `add`, `swiglu_quant` | **nenhuma mudança** — já são token-major | — |

`gate_quant` (na época `gate_mul`, antes de ser fundido com o `quantize_x`) merece nota:
com `n = attn_dim × n_tok`, o `h = i / head_dim` do shader avança
sozinho para `t * n_head + hh`, e `b_q` tem exatamente esse layout. Foi sorte da estrutura,
não projeto — mas está coberto pelo teste de ponta a ponta.

**A escrita no KV-cache** é um dispatch `KvPack` (cache em pares f16) que grava os `n_tok`
tokens do bloco de uma vez — as posições são consecutivas no cache.

**O prefill em batch está pronto** (buffers `n_batch×`, `decode_batch`, `plano_delta` com a
recorrência do bloco inteiro num dispatch). Sobre ele, o **verify** é um terceiro plano do
shard, `plan_verify`, com `VERIFY_TOK = 3` tokens fixos e logits de todos:

| peça | onde |
|---|---|
| escolha do plano por `Modo` (decode / batch / verify) | `build_plan`, `record_token` |
| `COLS = VERIFY_TOK` nos matvec | pipelines próprias, geometria do decode (`matvec_geom`) |
| logits dos três tokens | `b_logits` vira `vocab × 3`; **sem** requantizar |
| recorrência com pontos de snapshot | `plano_delta_verify` (cópia deliberada de `plano_delta`) |
| API | `ResidentForward::verify_shard`, `GpuResidentDecode::decode_verify` |

O "sem requantizar" é o detalhe que barateia o verify: o `norm_p2` da norma final já escreve
os três tokens em `b_xq`/`b_xd` no layout `t * n_blk + b`, que é exatamente o que o matvec de
três colunas lê. O custo extra de cada token é uma leitura a mais da ativação — o peso
Q6_K de 0,63 GB sai da VRAM uma vez só.

## Fase D — rollback do estado recorrente (PRONTA)

48 das 65 camadas são delta-net, **com estado recorrente**. Verificar `[t+1, t+2, t+3]`
avança o estado três passos; rejeitar uma proposta exige voltar um ou dois.

- Estado: `n_v_heads × d_state × head_v_dim × 4 B` = 3,1 MB por camada → **151 MB**, mais
  120 KB da janela da convolução por camada.
- O `plan_verify` copia estado e janela para dois buffers de snapshot (`snap[i]`, depois do
  token `i`) de cada camada linear (`PlannedOp::Copia`). `rollback_verify(manter)` copia de
  volta `snap[manter - 1]` a partir de command buffers gravados uma única vez na construção —
  a rejeição custa um submit.
- Salvar + restaurar a 717 GB/s: **~0,45 ms** projetados, ~1% do passo. **Ainda não medido.**
- O KV-cache não precisa de snapshot: as posições são consecutivas e o próximo passo
  sobrescreve as rejeitadas — basta recuar o comprimento.
- Os buffers de snapshot só existem com MTP ligado (`ResidentForward::new_shard_com`).

## Fase E — ligar e medir (LIGADA e medida)

`--mtp` no `llama-cli` e no `llama-server`, **desligado por padrão**. O laço de geração do
`llama-model` faz propor→verificar→aceitar/rollback; com sampling a proposta só é aceita se
coincidir com o token que o sampler tira dos logits[0], então a distribuição não muda.

No servidor o laço do `motor.rs` chama `Sessao::passo_mtp` e emite os aceitos por uma fila
de pendentes.

A taxa de aceitação da fase B (60,9 %) foi medida com uma proposta por token gerado. Em
produção a cabeça propõe uma vez por **passo**, e um passo aceito emite dois tokens — o
KV-cache da cabeça fica então uma posição atrás do do modelo a cada aceitação. É o mesmo
desenho do llama.cpp. Medida em geração real greedy com as duas propostas encadeadas:
**2,20 tokens por passo** (`greedy_com_mtp_gera_a_mesma_sequencia` imprime esse número).

O `docs/mtp-e-k80.md` registra **+3,4 %** para o MTP no llama.cpp neste mesmo modelo — fator
1,03 contra os 1,61 medidos aqui. A explicação mais provável é que a implementação dele não
faz batch eficiente num híbrido SSM; a cabeça do modelo prevê bem. Aquele número não deve
ser usado como estimativa do potencial.

**Conta do alvo:** com a base de hoje (26,3 tok/s a 2k) o MTP mede 34,4 tok/s; os 50 tok/s
pedem ainda uma base bem mais rápida — ver `docs/decode-por-configuracao.md`.
