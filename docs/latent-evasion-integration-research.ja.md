# Latent Evasion / CLE の Heretic 統合調査

調査日: 2026-09-27。実装前の調査。LLMでの性能・速度・VRAMは未測定。

対象は Heretic `3521f86`（調査開始時の作業ツリーは変更なし）、論文 [arXiv v2](https://arxiv.org/html/2605.21706v2)、公開実装 [pralab/latent-evasion @ 5f60ff9](https://github.com/pralab/latent-evasion/tree/5f60ff9eaa913be6b82585407de9c54d7809b220)。公開実装は `/tmp/heretic-latent-evasion-research` に取得して静的に確認した。

同タイトルは [NeurIPS 2026 公式一覧](https://neurips.cc/Downloads/2026) にも掲載されている。本調査の数式・実装比較は上記arXiv v2を対象とする。

## 判断

**統合は可能。忠実な比較基準として推論時のCLEを実装し、通常モデルとしての書き出しには、別工程でデータに基づく近似重みコンパイルを追加する構成が適切。**

追補: [SteerEditによる重みコンパイルの追加調査](/mnt/ssd1/heretic/docs/steeredit-weight-compilation-research.ja.md)。当初の保存に関する説明は、既存LoRAへの直接マージを対象としていた。較正データから新しい重み差分を求める方法なら、モデル構造を変えずにCLEを近似する書き出し経路を設計できる。性能は未検証。

| 対象 | 判断 | 主な条件 |
| --- | --- | --- |
| Hereticのモデル読込・バッチ処理 | 再利用可能 | 同じトークン列でプローブ学習・補正量計算・生成を行う |
| CLE-P | 統合可能 | Transformer block出力へのhookを追加 |
| CLE-A | 統合可能 | プロンプトごとの補正量を求める追加forwardと、その状態管理が必要 |
| Optuna・評価プラグイン | 再利用可能 | CLE用探索空間と、必要ならHarmBench judgeを追加 |
| 層ごとに異なる方向 | 対応可能 | CLE自体が層ごとに別のプローブを使う |
| 同じ層の複数方向 | 拡張可能 | 論文の標準CLEを超える研究拡張。非直交性・境界・符号の設計が必要 |
| 既存abliterationとの併用 | 実験可能 | 重み編集後の表現に対してプローブを再学習・検証する |
| 通常のLoRA/merged modelとしてCLEを保存 | 直接マージは不可。近似コンパイルは設計可能 | 較正工程・近似誤差評価を追加。忠実なruntime版も別途保持 |

ここで「多方向」は、層ごとに方向が違う場合と、同じ層に複数方向が存在する場合を区別する。特定の別ブランチや外部の多方向実装は指定されていないため、現在のcheckoutと一般的な複数方向の構成を対象にした。

## 現在のHereticとの対応

| 場所 | 確認した現状 | 統合に必要な変更 |
| --- | --- | --- |
| [main.py:587](/mnt/ssd1/heretic/src/heretic/main.py:587) | good/badの平均差から各層1本の方向を計算 | プローブ抽出・学習を方式別に分岐 |
| [main.py:618](/mnt/ssd1/heretic/src/heretic/main.py:618) | good平均に対する方向の直交化 | CLEのSVMに無条件で適用しない |
| [model.py:189](/mnt/ssd1/heretic/src/heretic/model.py:189) | 通常のLoRA rankは1 | 多方向の重み編集を追加するならrankも変更 |
| [model.py:461](/mnt/ssd1/heretic/src/heretic/model.py:461) | attention/MLPの出力行列をLoRAで変更 | CLEはblock出力への別の介入経路を使用 |
| [model.py:703](/mnt/ssd1/heretic/src/heretic/model.py:703) | hidden_statesの最後のprompt位置を取得 | block hookと同一の位置・正規化状態で採取 |
| [model.py:621](/mnt/ssd1/heretic/src/heretic/model.py:621) | 通常生成とlogit取得の共通経路 | CLEの準備・適用・後片付けを組み込む |
| [model.py:824](/mnt/ssd1/heretic/src/heretic/model.py:824) | chat用の別生成経路 | 同じ介入ランタイムを通す |
| [benchmark_score.py:45](/mnt/ssd1/heretic/src/heretic/scorers/benchmark_score.py:45) | HFLMに内部モデルを直接渡す | 特にCLE-Aの入力別準備を迂回しない設計が必要 |
| [model.py:315](/mnt/ssd1/heretic/src/heretic/model.py:315) | trial間でLoRAをリセット | hook・補正量・関連cacheもリセット |
| [main.py:1037](/mnt/ssd1/heretic/src/heretic/main.py:1037) | LoRA/merged modelを保存 | CLEの保存形式と再ロード経路を追加 |

`full_normalization_lora_rank` は正規化による重み差分の近似rankであり、意味的な方向数ではない。`direction_scope = "per layer"` も、同じ層の複数方向を意味しない。

現行プラグインは主に評価用で、公開Contextは介入APIを提供していない。CLE全体をScorerだけで実装すると、生成・chat・exportの整合性が崩れるため、Model側に介入の共通インターフェースを設ける。

## 公開コードの実際の動作

線形プローブを `s(h) = wᵀh + b` とすると、公開hookの変換は以下である。

```text
h' = h - β (wᵀh + b + m) / ||w||² · w
```

β=1なら `s(h') = -m`。m=0でもbが非ゼロなら原点を通る超平面への射影とは異なる。公開コードはReLUで条件分岐せず、すでに目標より負の側にある表現も目標の超平面まで動かす。CLI説明文にある「ReLU-gated」を実装仕様として採用しない。[公開hooks](https://github.com/pralab/latent-evasion/blob/5f60ff9eaa913be6b82585407de9c54d7809b220/utils/hooks.py)

- **CLE-P:** prefillとdecodeの全位置に射影hookを適用する。
- **CLE-A:** まず射影hookを付けてpromptをforwardする。各層の最後のprompt位置の差分をバッチ要素ごとに保存する。その後hookを外し、保存した差分を全位置へ加えるhookに切り替えてpromptから生成をやり直す。

CLE-Aの準備forwardでは、上流の介入が下流層に伝わる。また、準備forwardは全位置の射影、生成forwardは最後の位置から得た差分の一様加算であり、両者のprefillは同一ではない。無介入モデルから全層の差分を一括計算する簡略化や、準備forwardのKV cacheをそのまま生成に使う変更は、公開コードと一致しない。[CLE-A本体](https://github.com/pralab/latent-evasion/blob/5f60ff9eaa913be6b82585407de9c54d7809b220/cle-a.py)

## 再現性に影響する差・落とし穴

1. **プローブのスケール。** 論文Algorithm 1はwとbをwの元のノルムで正規化する。一方、公開trainerは生のLinearSVC係数を保存し、SVM loaderも正規化しない。hookの `||w||²` は未正規化wでの射影を扱えるが、同じ数値のmは同じ幾何学的距離にならない。既存artifactと同じ変換を維持して正規化するには、w・b・mをすべて同じノルムで割る。`paper_normalized` と `upstream_raw` の区別をメタデータに保存すべき。[trainer](https://github.com/pralab/latent-evasion/blob/5f60ff9eaa913be6b82585407de9c54d7809b220/classifier/train_latent.py)、[loader](https://github.com/pralab/latent-evasion/blob/5f60ff9eaa913be6b82585407de9c54d7809b220/utils/probes.py)
2. **採取点。** Hereticのhidden_statesはembeddingを含み、公開実装は `[1:]` で除く。さらにモデルによって最終hidden stateはfinal norm後で、最後のblock hookはnorm前になる。これは実モデルで照合が必要な潜在的不一致。独自学習では介入と同じblock出力から採取し、外部artifactでは採取方式を記録する。[公開表現抽出](https://github.com/pralab/latent-evasion/blob/5f60ff9eaa913be6b82585407de9c54d7809b220/models/language_models.py)
3. **学習表現の加工。** Hereticのwinsorization（標準では無効）や方向の直交化（標準では有効）を適用した空間と、介入時の生の表現は異なる。プローブの境界を変えずにそのまま流用しない。
4. **prompt形式。** Hereticにはresponse prefixの自動検出・思考部分を閉じる処理がある。公開モデルwrapperとsystem prompt/chat templateも一致するとは限らない。プローブは同一のテンプレート・prefix・model revision・量子化設定で学習する。論文再現時はHereticの追加prefixを明示的に制御する。
5. **学習ラベル。** 公開trainerはモデル別のfiltered JSONを読み、harmful/harmlessを1/0として学習する。その場でフィルタリングしているわけではない。Hereticの任意のgood/badデータを使う場合、実際の応答とラベルが一致するか別途確認する。
6. **探索の差。** 論文Appendix Dは層区間＋共通marginの探索後に層別marginを調整する。確認した公開 `optuna_search.py` は区間と共通marginの探索であり、後段の自動調整は見当たらない。またCLIのdataset既定値はtestなので、探索ではvalidationを明示する。[公開探索](https://github.com/pralab/latent-evasion/blob/5f60ff9eaa913be6b82585407de9c54d7809b220/optuna_search.py)
7. **採点。** Heretic標準の拒否キーワード率は、回答の成否を判定するASRと同じではない。KLは初回出力の分布差であり、能力保持全体の保証ではない。公開コードは探索用とtest用で別のHarmBench judgeを使う。[公開評価](https://github.com/pralab/latent-evasion/blob/5f60ff9eaa913be6b82585407de9c54d7809b220/utils/eval_jailbreaks.py)
8. **数値・状態管理。** スコアやノルム計算はFP32を基本とし、出力だけ元dtypeへ戻す設計が望ましい。ただし公開hookはhidden dtypeで計算するため、厳密なコード再現とは区別する。例外時のhook解除、空/ゼロノルムprobe、device_map、可変長batch、KV cache、batch変更を検証する。beam searchを追加するならCLE-Aの差分もbeamに合わせる。

論文上の正規化・学習ラベル・二段階探索の記述は [Algorithm 1 / Appendices C–D](https://arxiv.org/html/2605.21706v2) に基づく。公開コードとの差を吸収しただけで論文の数値まで再現できたとはみなさない。

## 多方向との組み合わせ

以下は今回の数学的検討・設計案であり、CLE論文で実証された結果ではない。論文はMDを比較対象として扱うが、CLEとMDを組み合わせた性能向上は保証していない。

### A. 各層で別々の1方向

もっとも容易。層lごとの `(w_l, b_l, m_l)` を保持する。Hereticの層取得は再利用できるが、CLEでは全層共通方向を別層へ移植する必要はない。

### B. 直交するK方向を同じ層で扱う

列が正規直交する行列 `Q ∈ R^(d×K)` を用意し、各軸の切片bと目標margin mを定義すれば、同時変換を以下に拡張できる。

```text
h' = h - Q (Qᵀh + b + m)
Qᵀh' + b = -m
```

b=m=0なら通常の部分空間除去 `h'=(I-QQᵀ)h`。各軸の符号をharmful側が正になるようそろえ、bを較正する必要がある。PCA/SVD/SOMから得た基底は、そのままで各軸が拒否を予測するとは限らない。軸ごとに正のmarginを指定する根拠を検証する。

### C. 非直交な複数プローブ

法線を列に並べた `U ∈ R^(d×K)` が列フルランクなら、すべての等式境界 `Uᵀh'+b=-m` を満たす最小L2変化は以下。

```text
h' = h - U (UᵀU)^(-1) (Uᵀh + b + m)
```

実装では逆行列の明示計算を避け、solve/QR/SVDを使う。`UᵀU≈I` を仮定して単純加算すると、他方向のスコアを動かして目標を外す。順番に射影しても非直交なら順序依存になり、一巡で同時制約を満たす保証はない。

rank不足の場合、擬似逆行列は使えるが、矛盾した目標を同時に達成することはできない。残差・条件数を検証する。QRで基底だけ取り替え、元のbやmをそのまま流用するのも不可。

「各境界の負側に入ればよい」とする場合は、等式への射影ではなく `Uᵀh'+b≤-m` を満たす最小距離問題として設計できる。ただしこれは公開CLEとは別の介入方式で、制約の実行可能性と解法を追加検証する必要がある。

### D. 多方向部分空間内で1本のSVMを学習

K次元の `Qᵀh` でSVMを学習し、係数aから `w=Qa` を作れば、部分空間を使って方向を推定できる。ただし最終的な境界の法線は1本であり、「K方向に同時介入」とは区別する。比較対象として有用。

### E. 多方向の重み編集後にCLEを適用

実行は可能だが、事前学習したCLEプローブの分布がずれる。まず多方向重み編集を固定し、そのモデルからプローブを再学習してCLE単独・多方向単独・併用を比較する。方向除去によって残った信号が少なくなり、追加効果が出ない可能性もある。

方向選択・層・margin・強度を一度に探索すると比較しにくい。まず方向集合を固定し、CLEの効果を測定してから探索対象を増やす。

## 重みへのマージと保存

正規化なしの多方向重み編集なら `ΔW=-QΛQᵀW` をrank K以下のLoRAで表現できる。しかし、これはblock全体に対するCLEではない。

CLE-Pはblock出力に対して線形射影と定数シフトを適用する。block出力にはskip connectionも含まれ、現行のattention/MLP行列へのLoRAだけでは同じ変換を表現できない。さらにCLE-Aは補正量が入力に依存するため、全入力で共有する固定LoRAへの単純マージはできない。

忠実なruntime版の保存形式はbase model参照と、probe tensor、介入設定、採取点、正規化方式、テンプレート、seed、データrevision、コードrevisionを含む研究用artifactとする。別工程で近似コンパイルした重み差分をマージできれば、通常の `save_pretrained()` による標準モデル保存が可能。これはruntime版と同一の全入力動作を保証するものではない。詳細は上記の追補を参照。

## 依存関係と実装単位

公開実装はTransformers 4.57.6、Hereticは5.6系を指定している。PEFTやhuggingface-hubなどの指定も異なる。公開requirements全体を既存環境に入れる方法は避け、数式と必要な処理をHereticの現行APIへ移植する。[公開requirements](https://github.com/pralab/latent-evasion/blob/5f60ff9eaa913be6b82585407de9c54d7809b220/requirements.txt)、[Heretic依存](/mnt/ssd1/heretic/pyproject.toml:27)

SVMに必要なscikit-learnはHereticのresearch extraに既に存在する。judgeの実行環境は生成モデルと分離する構成も可能。公開コードはMIT表記で、コードを転用する場合はPRALabの著作権・ライセンス表示を保持する。[公開LICENSE](https://github.com/pralab/latent-evasion/blob/5f60ff9eaa913be6b82585407de9c54d7809b220/LICENSE)

推奨する実装順序（以下のファイル名・設定名は提案で、まだ追加していない）:

1. `probes.py`: block出力の採取、DiM/SVM、artifact形式、モデル・層・次元・採取点検証。
2. `interventions.py`: CLE-Pのcontext managerと解除保証。Modelの通常生成・logit・chatから共通利用。
3. CLE-A: 公開コードに合わせた準備forward、一様加算、入力単位の状態管理。HFLMのteacher-forcingも含めた入力経路の設計。
4. `config.py` / `main.py`: `abliteration / cle_p / cle_a` の方式選択、方式別探索、trial復元。既存方式を既定として維持。
5. 評価・保存: HarmBench scorer、validation/test分離、runtime artifactのsave/load、reproduce schemaの拡張。
6. 同一層のK方向: まずrank=1への一致を確認し、固定した小さなKから比較する。

プローブ学習には平均だけでなく個々の表現が必要。FP32でN=256、L=32、d=4096なら表現だけで約128 MiB（モデル・一時領域は別）。CPUへ逐次offloadする設計が可能。CLE-Aは通常生成のprefillに加えて準備用prompt forwardが必要で、生成全体が一律2倍になるわけではない。具体的なコストはモデル・prompt長・生成長で測定する。

## 今回行った検証と次段階の判定基準

既存 `.venv` のPyTorch 2.13.0+cu130を使い、CPU上の小さなtensorで以下を確認した。モデルのダウンロード・GPU推論・依存更新は行っていない。

- 公開 `projection_hook` のβ=1でスコアが `-m` になる。
- 公開CLE-A hookの保存差分が最後のprompt位置の差分に一致する。
- 準備時の全位置射影と、保存差分の一様加算は最後の位置以外では一般に異なる。
- w・b・mを同じノルムで割れば公開hookの結果が一致する。
- 非直交2方向の数値例で、単純加算の目標スコア誤差は2.76、同時解は約8.9e-16。

これらは数式と公開hookの局所検証であり、実モデルでの動作・性能検証ではない。

次段階では、同一モデル・データ・予算で「無介入、現行Heretic、CLE-P、CLE-A、多方向単独、多方向+CLE」を比較する。採用条件は、ASRだけでなく、harmless側のKL、能力ベンチマーク、過剰拒否、出力崩壊、速度・VRAMを含めて判断する。

再現性の必須検証は、公式rank=1 hookとの一致、β=0の恒等動作、batch/単独推論の一致、最終層採取点の一致、hook解除後のbaseline復元、cache有無、save/load前後の一致、rank不足・方向の符号・marginのスケールである。CLE-AでHFLMが準備処理を迂回した状態のベンチマークを、CLE-Aの能力評価として扱わない。
