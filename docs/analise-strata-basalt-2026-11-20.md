# O que o Strata e o Basalt têm que servem no MI50

**Data:** 2026-11-20 · **Motivo:** levantar o que os dois motores especializados já resolveram e
ver o que é técnica transferível para este runtime, em vez de adivinhar.

Regra de triagem usada aqui: entra o que é **técnica**. Sai o que é propriedade do modelo deles.
Expert cache, NVFP4, `resident-experts` e os drafters DFlash2 (Strata) e DSpark (Basalt)
pertencem aos MoE deles.

Alvos de hardware, conferidos no GitHub em 2026-11-20:
- **Strata** roda em NVIDIA RTX 20–50, AMD RDNA2–4, Intel Arc, Apple Silicon, Strix Halo e DGX
  Spark. Tem build da comunidade para gfx906 (`-DSTRATA_HIP_GFX906=ON`, `docs/AMD_HIP.md`).
  Modelos: Qwen3.8-Flash-Next e Qwen3.6-35B-A3B.
- **Basalt** é só NVIDIA Blackwell (sm_120, CUDA 13). Não tem HIP, Vulkan nem ROCm.

## Fontes consultadas

| Projeto | Onde | O que foi lido |
|---|---|---|
| Strata (`Niko1221/Strata`) | GitHub | `README.md`, `docs/HOW_IT_WORKS.md`, `docs/DETAILS.md`, árvore de `engine/src/` e `engine/include/` |
| Basalt (`jesdga95/basalt`) | GitHub | `README.md`, `docs/configuration/index.md`, `src/spec/suffix_drafter.cpp`, `src/spec/draft_policy.cpp`, árvores de `src/` |

Nada foi copiado de código deles neste documento. São apontamentos de onde olhar.

## Resumo das seis frentes

| # | Frente | Origem | Estado no MI50 hoje | Esforço | Ganho esperado |
|---|---|---|---|---|---|
| 1 | Vários checkpoints de conversa | llama.cpp, `basalt src/core/conversation_store.cpp` | Um snapshot só, na fronteira do turno | 1–2 dias | TTFT continua caindo depois dos 31,5 s |
| 2 | Drafting por sufixo (prompt lookup) | Strata; `basalt src/spec/suffix_drafter.cpp` | Não existe | 2–3 dias | 5–10% no decode de código, zero no texto novo |
| 3 | Decisão de especulação por medição | `basalt src/spec/draft_policy.cpp` | Corte fixo `LLAMA_RS_MTP_ATE=12k`; adaptativo já tentado e descartado | ½–1 dia | Incerto: só se resolver o viés que derrubou a tentativa de 2026-09-26 |
| 4 | Prefill em blocos grandes | Strata (8.192 por vez); MMQ vendorizado do llama.cpp | Bloco 256/32, tile de 32 colunas | 2–4 dias (tile) · semanas (delta-net) | Ataca a maior lacuna: 171,9 contra 318 tok/s frio |
| 5 | KV em int8 | Basalt `"kv": "int8"`; Strata 8-bit KV acima de 4K | f16 empacotado | 1–2 dias | Capacidade (2,2 GB → 1,1 GB em 32k), não velocidade |
| 6 | Produto: endpoint Anthropic e modos de energia | Ambos falam OpenAI **e** Anthropic; Basalt tem `lazy`/`idle-unload`/`idle-sleep` | Só API OpenAI; 19 GB de pesos sempre residentes | 1 dia (endpoint) · 2–3 dias (energia) | Servir Claude Code; devolver VRAM ao jogar |

Ordem sugerida: 1, 2, 3, 4, 5, 6.

## 1. Vários checkpoints de conversa

O Strata guarda a conversa e processa só o que é novo. O llama.cpp guarda **vários** checkpoints.
O Basalt tem um `conversation_store.cpp` de 70 KB e um `checkpoint_transfer.cpp`. Ele guarda em
disco, abaixo do cache em RAM, os ids dos tokens, os comprimentos dos checkpoints, blocos de K/V e
o estado recorrente do DeltaNet. O intervalo é a chave `prompt-cache-every`, padrão 16384 tokens.

O MI50 já fez o primeiro passo: um snapshot depois do último token especial, e o TTFT de 30k caiu
de 387 s para 31,5 s. Falta o segundo: um checkpoint a cada N tokens da conversa. Hoje uma
divergência antes do snapshot reprocessa o prompt inteiro. Com checkpoints a cada N tokens, custa
no máximo N.

Dificuldade real: as camadas delta-net têm estado recorrente, que não volta atrás como o KV de
atenção. Um checkpoint tem de incluir esse estado, não só o KV e o comprimento. O Basalt faz
exatamente isso.

## 2. Drafting por sufixo

É o "prompt lookup" do Strata: até 5 tokens copiados do próprio texto anterior (até 12 na build
CUDA de 8 linhas, com +11 a +14% em reescritas). O Strata mediu 6–11% mais rápido em edições de
código, sem mudança em outro texto, e só liga onde a aceitação medida diz que vale.

O Basalt tem o código pronto em `src/spec/suffix_drafter.cpp`, 106 linhas: duas tabelas hash
(uma de trigrama, uma de ordem longa = 8), 4 posições por chave, o casamento mais longo vence, e o
forward verifica. Custa uma consulta de hash por passo, quase nada frente a um decode.

É exatamente o caso de uso de um agente: o modelo reescreve texto que já está no contexto. E
combina com o MTP: nos trechos que repetem o contexto, o sufixo manda mais que as 2 propostas do
MTP.

Cuidado: é um segundo drafter com custo próprio. A verificação é o forward que você já tem, então
o custo extra é só a atenção sobre as propostas, igual ao verify do MTP.

## 3. Decisão de especulação por medição

O `draft_policy.cpp` do Basalt (255 linhas) mantém um custo por tamanho de janela em EMA, uma EMA
de aceitação para o MTP, contagens com decaimento e priors por faixa de casamento para o lookup, e
3 probes antes de vetar um tamanho.

O MI50 tem um corte fixo: `LLAMA_RS_MTP_ATE=12k`, que já recuperou a perda acima de 12k. A
decisão 11 (2026-09-26) tentou uma escolha adaptativa e descartou: a cabeça MTP tem KV próprio,
que não anda nos passos simples, e as provas de MTP saíam com aceitação menor. A estimativa
favorecia o passo simples por construção. Portar a política do Basalt só vale se resolver esse
viés.

## 4. Prefill em blocos grandes

O Strata lê 8.192 tokens por vez e passa de 1.000 tok/s em texto longo. O MI50 está em 256, a
171,9 tok/s frio contra 318 do llama.cpp HIP.

O próximo degrau não é o bloco, é o **tile**. O MMQ do Strata é a cópia do llama.cpp
(`engine/third_party/strata-llama/ggml/src/ggml-cuda/mmq.cuh`). O tile sai de uma tabela por
arquitetura e por tipo: 64 ou 128 linhas, de 8 a 128 colunas. Na tabela GCN, o Q4_K tem entre as
opções 128×64. O grid põe os tiles de linha em X e os de coluna em Y. O MI50 já tem 32 colunas e o
bloco 256. Subir para 64 colunas e depois 512 de bloco é o caminho mais curto.

O trabalho pesado é o delta-net em batch: o `docs/prefill-em-batch.md` descreve o laço sobre os
tokens dentro do kernel, estado em registrador entre eles. É o que torna o bloco grande viável.
São 48 camadas de recorrência.

Itens já mapeados no backlog e que seguem valendo: split-K nas saídas estreitas (item 1.3) e
estudar o MMQ do HIP (item 1.4).

## 5. KV em int8, mas pela razão certa

O Basalt usa `"kv": "int8"` e o Strata liga 8-bit KV acima de 4K. No MI50 o ganho é de
**capacidade**, não de velocidade. O KV f16 empacotado cortou a VRAM pela metade e não mudou o
tempo da atenção, e a decisão 12 mostrou que ela não é limitada por conta nem por bytes do KV.

Use int8 para caber contexto ou sobrar VRAM, não espere tok/s.

Antes de comparar builds, fixe o split com `LLAMA_RS_SPLIT`: a divisão automática varia entre
execuções e move o resultado ~2%.

## 6. Produto

O endpoint `/v1/messages` da API Anthropic. O Strata e o Basalt falam as duas APIs; o MI50 só
fala a OpenAI. Um dia de trabalho e o motor passa a servir o Claude Code sem tradução. O Codex
fala a API da OpenAI e não precisa desse endpoint.

Depois valem os modos de energia do Basalt (`lazy`, `idle-unload`, `idle-sleep`), que devolvem
VRAM quando se joga. Hoje os 19 GB de pesos ficam residentes o tempo todo.

## O que não portar

- **Streaming de KV pela RAM** (Strata faz isso a partir de 64K). São 16 GB/s de PCIe contra
  836 GB/s medidos de HBM2. Num modelo denso o KV quente tem de ficar em VRAM. Só tente se medir.
- **Tabela n-gram de 28,8 GB.** É um arquivo de dados no SSD, lido algumas linhas por token;
  `engine/src/ngram` só tem o leitor. É artefato do modelo, não técnica.
- **Expert cache, tiers adaptativos, `resident-experts`, peer experts.** Tudo para MoE, que o
  llama.rs não suporta.
- **Kernels fp8/fp4 e drafters DSpark/DFlash2.** Hardware e modelos específicos.

## Validação

Conferido no GitHub (HEAD de `main`) em 2026-11-20:

1. Strata: 8-bit KV acima de 4K, KV streaming a partir de 64K, blocos de 8.192 tokens e 1.000+
   tok/s em texto longo conferem (`docs/DETAILS.md:45`, `docs/HOW_IT_WORKS.md:50-51`). Na gfx12 o
   `--prefill auto` usa blocos de 32.768.
2. Basalt: `"kv": "int8"` opcional (padrão fp16), `suffix_drafter.cpp` e `draft_policy.cpp`
   conferem, com os números corrigidos acima.
3. Os dois falam a API Anthropic (README do Strata, linha 135; README do Basalt, linha 5).
4. `conversation_store.cpp`: 70.159 bytes, 1.467 linhas; guarda o estado do DeltaNet.

Ainda são palpite, não medição:
- Os esforços estimados.
- O ganho de 5–10% do drafting por sufixo, que é o número do Strata para edições de código.
- A ordem 1→6. Ela pode mudar se o objetivo for outro, por exemplo vencer o prefill.
