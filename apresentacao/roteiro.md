# Roteiro — Vídeo STAR (Tech Challenge Fase 4)

Duração alvo: **~5 minutos**. 10 slides, ~30s cada. Fale no ritmo natural, sem pressa — o roteiro já está curto de propósito.

---

## Slide 1 — Título (15s)

> "Olá! Neste vídeo vou apresentar o projeto que desenvolvi para a Fase 4 do Tech Challenge: uma camada de **Sustentação e Confiabilidade** para um modelo de Credit Scoring em produção — ou seja, como garantir que o modelo continue confiável depois do deploy."

---

## Slide 2 — Situação (40s)

> "Vamos começar pela situação. Um modelo de crédito é treinado, validado, e colocado em produção com uma boa performance. Mas o mundo real não é estático: a inflação muda a renda das pessoas, políticas de crédito mudam, o perfil de quem pede empréstimo muda.
>
> Isso gera dois problemas distintos: **Data Drift**, quando a distribuição dos dados de entrada muda, e **Concept Drift**, quando a própria relação entre essas variáveis e o risco de inadimplência muda.
>
> Sem contratos de dados, sem monitoramento e sem governança, essa degradação é silenciosa — só é percebida quando o prejuízo já aconteceu."

---

## Slide 3 — Tarefa (30s)

> "A tarefa era construir exatamente essa camada de sustentação, em quatro pilares: contratos de dados para validar cada lote antes de alimentar o modelo; simulação e detecção de drift de dados e de conceito; observabilidade de todo o pipeline; e documentação de governança e LGPD."

---

## Slide 4 — Ação: Arquitetura (35s)

> "Para a ação, construí um pipeline único e testável. Ele parte da base real do Lending Club, passa por um contrato de dados em Pandera, treina um modelo baseline em XGBoost, simula seis lotes de produção com drift crescente, detecta esse drift estatisticamente com PSI e Evidently, e registra tudo em observabilidade com MLflow e um dashboard.
>
> Todo esse fluxo é coberto por 18 testes automatizados com pytest, todos passando."

---

## Slide 5 — Ação: Contratos de Dados (35s)

> "O contrato de dados, feito com Pandera, tem 17 regras de validação: faixas numéricas, domínios categóricos, checagem de nulos e de duplicatas.
>
> Para provar que ele funciona, simulei um lote sujo de propósito — com valores nulos, renda negativa, FICO inválido e duplicatas. O contrato identificou 24 violações em 3 colunas, bloqueou o lote e o enviou para quarentena, com o evento registrado no log do pipeline."

---

## Slide 6 — Ação: Drift (45s)

> "Para o drift, simulei seis lotes de produção. No Data Drift, fui deslocando gradualmente a renda, o DTI e a taxa de juros. No Concept Drift, fiz com que perfis antes considerados bons passassem a ter maior chance real de inadimplência — simulando o próprio cenário de inflação que mostrei na situação.
>
> Um ponto importante: eu decidi o alerta de drift usando **PSI**, o Population Stability Index, em vez de testes de hipótese como KS ou qui-quadrado. Isso porque, em amostras grandes, esses testes acusam drift em quase tudo, mesmo sem mudança relevante — o PSI mede o tamanho real do efeito.
>
> O resultado: no lote zero, sem drift, zero features são sinalizadas e o AUC fica estável. Conforme o drift aumenta, a AUC cai de forma consistente e o número de features sinalizadas sobe — exatamente o comportamento esperado."

---

## Slide 7 — Ação: Observabilidade (30s)

> "Na observabilidade, cada etapa do pipeline grava um log estruturado em JSON. O MLflow rastreia o treino do baseline e cada lote de drift analisado, com métricas e alertas. E um dashboard em HTML consolida tudo isso visualmente: AUC por lote, taxa de default, features em drift e status dos eventos do pipeline."

---

## Slide 8 — Resultado (30s)

> "Como resultado, temos 18 de 18 testes passando, um AUC baseline de 0,73, uma queda de até 0,07 de AUC no lote de maior drift, e exatamente as features que eu alterei de propósito sendo sinalizadas pelo PSI — nenhum falso positivo no lote limpo.
>
> Todo o código está no GitHub, com histórico de 10 commits semânticos, cada um correspondendo a uma etapa validada e testada."

---

## Slide 9 — Governança & LGPD (30s)

> "Na parte de governança, documentei os princípios de Privacy by Design: colunas de identificação nunca são carregadas, a base legal usada é a de execução de contrato e legítimo interesse, há previsão de revisão de decisões automatizadas, e cuidado para não usar variáveis como proxy de localização ou renda que possam introduzir viés.
>
> Também documentei a cadeia causal: inflação sobe a renda nominal, mas não o poder de compra real, o que eleva o DTI e os juros, tornando perfis antes bons genuinamente mais arriscados — isso é concept drift, não só um artefato estatístico."

---

## Slide 10 — Conclusão (20s)

> "Em resumo: construí um pipeline único que integra validação, simulação e detecção de drift, observabilidade e governança — testado ponta a ponta e pronto para sustentar um modelo de crédito real em produção. Obrigado!"

---

## Dicas de gravação

- Grave em um ambiente silencioso, de preferência com fone de ouvido com microfone.
- Pratique uma vez em voz alta antes de gravar, cronometrando — o texto acima soma ~5 minutos em ritmo normal.
- Ao importar no Google Slides: `Arquivo → Importar slides` ou simplesmente abra o `.pptx` diretamente no Google Slides (ele converte automaticamente).
- Use a gravação de tela do próprio Google Slides/Meet, ou OBS Studio, em modo apresentação.
