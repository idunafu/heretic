# SOM Directions の Heretic 統合調査

調査日: 2026-09-29。結論は **統合可能であり、PoC より改善して実装する具体的な道筋がある。ただし性能向上は実モデルで未検証**。既存 PR #196 を移植するより、現在開発中の modifier plugin に合わせて SOMA を実装する方が保守しやすい。

今回の範囲は論文・公開コード・PR の最終差分と議論・現在の checkout の調査、および CPU 上の小さな行列による検証。本体の実装変更、モデルのダウンロード、GPU での探索、実モデルのベンチマークは実施していない。

確認した版:

| 対象 | 版 |
| --- | --- |
| 論文 | [arXiv:2511.08379v2](https://arxiv.org/html/2511.08379v2)、AAAI 2026 採択 |
| 参考実装 | [pralab/som-refusal-directions](https://github.com/pralab/som-refusal-directions/tree/d244c7d282ac65a1520bef0d418615ef148108af)、`d244c7d` |
| 既存 PoC | [PR #196](https://github.com/p-e-w/heretic/pull/196)、head `444edece46741c6eb92cf7a9fe33a2929384cce8` |
| ローカル / upstream master | `3521f8648a0dccf6e12a92666862632235fac7e6`、`2.0.0.dev0`。調査時点で双方一致 |
| 新しい統合基盤 | [PR #446](https://github.com/p-e-w/heretic/pull/446)、head `ce8b05c77b82f0b350e137d7b783a2eb5e1059db`、open / draft / 未マージ |

**最も重要な更新は、統合先となる API がすでに PR #446 にあること。** #196 は 2026-07-07 に作者がプラグイン化を理由に close した。メンテナは同日、modifier plugin を ARA と SOMA の双方に使えるよう設計し、ARA の後に SOMA を統合する方針を示した。手法が失敗したための却下ではない。[close コメント](https://github.com/p-e-w/heretic/pull/196#issuecomment-4903433697)、[メンテナの方針](https://github.com/p-e-w/heretic/pull/196#issuecomment-4905192746)

9 月に作られた #446 はその modifier plugin と ARA の実装で、#211 を置き換え、#196 も復活させると明記している。現在の master は scorer plugin までで、modifier plugin はまだ使えない。したがって「現在すでにプラグインとして差し込める」とは言えないが、API を独自に一から作る必要性は薄れた。#446 は draft なので、その head に依存する開発では API 変更を見込む。[PR #446](https://github.com/p-e-w/heretic/pull/446)

**論文の中心は、bad 表現から SOM で複数の代表点を学び、各代表点から good 平均を引いて候補方向を作ること。** 単一層で候補を作り、方向の選択と順序を探索する。公開実装は、選んだ方向を順に重みへ適用している。これは、各方向の更新を元の重み上で計算して足し合わせる #196 と同じ演算ではない。[論文 Algorithm 1 / Appendix B](https://arxiv.org/html/2511.08379v2)、[公開探索](https://github.com/pralab/som-refusal-directions/blob/d244c7d282ac65a1520bef0d418615ef148108af/optuna_search.py#L97)

| 論点 | 論文・参考実装 | #196 の最終コード | 推奨する扱い |
| --- | --- | --- | --- |
| 候補抽出 | 選んだ 1 層で SOM。good は平均 | embedding を含む各 hidden-state slot で SOM | 初期版は source layer を離散指定・選択し、複雑化を抑える |
| 方向選択 | 候補から k 本を探索 | top-k を事前固定し重みを探索 | 候補生成と選択を分離 |
| 多方向の合成 | 順序付き逐次更新 | 元 W に対する差分の和 | `sequential` と `additive` を別方式として保存・評価 |
| embedding | 公開コードは embedding も編集 | attention / MLP 出力行列のみ | Heretic 互換版と公開コード再現版を区別 |
| LoRA | 公開コードは直接重み更新 | none/pre は rank 1、full は別設定の rank | none/pre は k 以下の差分を直接因数分解 |
| 評価 | judge による ASR | 主に拒否キーワード数と KL | 保持した test set、judge、能力評価を追加 |
| 保存 | 編集済みモデルを利用 | Heretic の保存経路 | adapter / merge / 再ロード / 再現情報まで確認 |

参考実装の source layer は `hidden_states[1:]` の添字。Heretic は embedding slot を含むため、同じ整数をそのまま使うと 1 slot ずれる。最終 hidden state の正規化位置、chat template、system prompt、response/thinking prefix も一致させる必要がある。[参考実装の採取](https://github.com/pralab/som-refusal-directions/blob/d244c7d282ac65a1520bef0d418615ef148108af/models/language_models.py#L186)、[現在の採取](/mnt/ssd1/heretic/src/heretic/model.py:703)

**#196 に残っている改善点は、コードから確認できる。** 以下は古いレビューコメントの転記ではなく、最終 head を調べた結果。

1. **top-k の頻度を測る対象が違う。** `get_top_k_neuron_weights()` は学習サンプルではなく `self.som._weights` 自身を `winner()` に渡している。異なる prototype は通常自分自身が winner になり、ほぼ全員 1 票になる。これでは bad prompt の支持数を表さない。実データの BMU 割当を数え、安定した neuron ID と支持数を保存する。なお、支持数上位を選ぶだけでも論文の BO 選択とは違う。[som.py:70](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/som.py#L70)

2. **none/pre では多方向の差分を rank 1 に潰している。** LoRA 初期化が rank 1 のままで、合算した差分を SVD で rank 1 に切り詰める。方向数 k を設定しても k 方向の更新が保持されない。FULL の rank は `full_normalization_lora_rank` で別に決まり、方向数とは別概念。候補生成が変わる効果は残るので、報告された改善そのものが否定されるわけではない。[rank 初期化](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/model.py#L173)、[SVD 部分](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/model.py#L502)

3. **探索しても使われない `min_weights` がある。** 層の強度は `max_weights[0]` と `min_weights[0]` から作り、各方向には `max_weights[i] / max_weights[0]` を掛ける。そのため `min_weights[1:]` は値を変えても更新に影響しない。方向ごとの独立な下限を意図するなら式を修正し、共通の減衰を意図するなら不要な探索パラメータを削除する。[強度計算](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/model.py#L443)、[方向ごとの倍率](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/model.py#L483)

4. **非直交方向の和と逐次適用を混同している。** #196 は各差分を同じ W から計算して足す。参考実装は前の方向で変更した W に次を適用する。方向間の相関が高いと差は大きい。また #196 は不足方向を複製して埋めるため、additive では同じ方向への強度が増える。複製による穴埋めをやめ、有効候補数・ゼロノルム・重複を明示的に扱う。[合算処理](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/model.py#L475)、[複製処理](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/main.py#L456)

5. **層間の方向 ID に対応付けがない。** 別々に学習した SOM の同じ slot / 順位は、同じ概念を表す保証がない。そのまま隣接層の方向を補間している。global は整数の source layer を選び、連続補間をしない。補間を追加する場合は、同じ prompt の割当や方向類似度で対応付けを検証する。一方、各層で抽出した方向をその層に適用する per layer は補間を必要とせず、この理由で除外する必要はない。現在は per layer と両方式の自動選択も実装済み（[使い方](som-module.ja.md)）。[補間処理](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/model.py#L400)

6. **探索次元・再現性・依存管理を整理できる。** 方向ごとに min/max を増やす設計は、効果のない変数も含め TPE の探索を広げる。SOM の seed は固定 0、SVD は局所的な再 seed がなく、`minisom` は `pyproject.toml` の依存に追加されていない。設定 seed、SOM 版、候補 ID、選択順、保存済み方向の hash を管理する。none/pre の SVD 廃止は速度・メモリだけでなく RNG 依存も減らせる。[SOM 初期化](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/som.py#L44)、[依存定義](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/pyproject.toml)

過去に報告された非 SOM 経路の tensor 軸の不具合は、最終 head では `unsqueeze(0)` に修正されている。また、古いレビューにある「base weight を直接書き換えて reset を壊す」という指摘も、最終版の none/pre では LoRA への代入に変更されている。これらを未修正の問題として数えない。[最終 main.py](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/main.py#L471)、[最終 adapter 代入](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/model.py#L551)

**改善の中核は、多方向の演算と LoRA の表現を分けること。** 以下の式は今回の導出。W は出力 projection の重み、v_i は単位方向、λ_i は強度とする。

additive では次の差分を rank k 以下でそのまま表せる。

```text
V = [v₁ … vₖ]
ΔW = −V diag(λ₁ … λₖ) Vᵀ W
B = −V diag(λ₁ … λₖ)
A = Vᵀ W
ΔW = BA
```

巨大な `total_delta_W` を作って SVD する必要はない。元の非直交方向を正規直交化して `QQᵀ` に置き換えるのは、単なる実装上の最適化ではなく別の演算になる。内部計算に QR 等を使う場合も、係数を保持して元の operator を再現する必要がある。

sequential も、同じく rank k 以下で表せる。

```text
P_i = I − λ_i v_i v_iᵀ
W_k = P_k … P_2 P_1 W
ΔW = W_k − W
```

途中までの更新を `W + BA` として保持すれば、次の方向について

```text
a_iᵀ = v_iᵀ W + (v_iᵀ B) A
b_i  = −λ_i v_i
B ← [B, b_i]
A ← [A; a_iᵀ]
```

と追加するだけで、逐次的な重み更新と同じ差分になる。差分の rank は k 以下だが、順序は保持する。非直交の 2 方向なら逐次式には `λ₂λ₁ v₂(v₂ᵀv₁)v₁ᵀW` という交差項があり、単純和には存在しない。

この「厳密」は FP の丸め誤差を除いた、指定された**重み更新**との代数的な同値を指す。論文の抽象的な block 出力への介入と、全 architecture・全 residual 経路で等価だと主張するものではない。参考実装に合わせるなら embedding の編集も別途必要で、weight tying や projection 後の normalization も確認する。[公開重み更新](https://github.com/pralab/som-refusal-directions/blob/d244c7d282ac65a1520bef0d418615ef148108af/utils/ablation_utils.py#L4)、[モデル別の編集対象](https://github.com/pralab/som-refusal-directions/blob/d244c7d282ac65a1520bef0d418615ef148108af/models/language_models.py#L290)

PRE では、元の行ノルムを対角行列 D とし、行正規化した W̃ に対して因数を計算して、B 側に D を掛ければよい。ゼロ行は既存の正規化規則に合わせて扱う。これも rank k 以下を保つ。

FULL では、編集後の各行を再正規化するため、最終差分は一般に rank k 以下ではない。ここは SVD 等の近似を残し、`direction_count` と `adapter_rank` を別設定にする。近似の相対誤差と行ノルム誤差も測る。**現在の master の既定値は FULL / rank 3、#196 の既定値は NONE** なので、同じ「デフォルト同士」でも比較条件が異なる。[現在の設定](/mnt/ssd1/heretic/src/heretic/config.py:371)、[PoC の設定](https://github.com/kabachuha/heretic/blob/444edece46741c6eb92cf7a9fe33a2929384cce8/src/heretic/config.py#L187)

**CPU での確認結果。** 取得した #196 の最終ソースから対象メソッドを AST で取り出し、小さい模擬 module / SOM に適用した。モデル・MiniSom の学習・PEFT 全体を使う統合試験ではない。数式の確認は NumPy float64、PoC の adapter 計算は CPU PyTorch float32。

| 確認項目 | 結果 |
| --- | --- |
| 学習データの多数派が prototype `[3,3]` となる例 | PoC の top-1 は `[0,0]`。自己投票と実データの支持数が一致しない |
| PoC の `min_weights[1:]` のみ変更 | 得られた差分の最大絶対差 `0.0` |
| 3 方向の additive 差分を PoC の NONE 経路で処理 | 意図した差分 rank 3 → adapter 差分 rank 1 |
| 上記の rank 1 近似の相対 Frobenius 誤差 | `0.67345`。この toy 例の値であり実モデル性能の低下率ではない |
| additive の直接 LoRA 因数と dense 計算の最大誤差 | `2.22 × 10⁻¹⁶` |
| sequential の直接 LoRA 因数と dense 逐次計算の最大誤差 | `2.22 × 10⁻¹⁶` |
| PRE + sequential の同じ比較 | `3.61 × 10⁻¹⁶` |
| 8×11 の重みを 3 方向で更新し FULL 正規化 | 最終差分 rank 8。rank 3 以下とは限らない |

同じ例で additive と sequential、sequential の順序を逆転した結果も一致しなかった。検証時の環境は NumPy 2.3.5 / PyTorch 2.13.0+cu130。作業用の [検証スクリプト](/tmp/heretic-som-research/check_som.py) と [結果 JSON](/tmp/heretic-som-research/checks.json) を `/tmp/heretic-som-research` に保存した。一時ファイルなので長期保存する研究 artifact とは区別する。

**実装先は #446 の SOMA modifier が適切。** 現 master の scorer から model を書き換える設計では、trial reset、評価、chat、export の責任分担が崩れる。#446 は必要な接続点を用意している。[Modifier API](https://github.com/p-e-w/heretic/blob/ce8b05c77b82f0b350e137d7b783a2eb5e1059db/src/heretic/modifier.py#L38)

| 接続点 | SOMA での用途 |
| --- | --- |
| `init(ctx)` | prompt 読込、good 平均、bad 表現、SOM 候補を一度だけ計算 |
| `ctx.get_model()` | model を公開 API から取得 |
| `model.apply_lora(rank)` | 必要な rank で初期化 |
| `suggest_parameters(ctx, trial)` | 候補 ID、順序、共通強度、層・component の設定を提案 |
| `modify_model(ctx, parameters)` | 選んだ operator の A/B を構成 |
| `reset_model(ctx)` | trial 間の adapter 初期化、merge 後の再ロード処理 |
| `Parameters.to_dict/from_dict` | trial 復元と再現用シリアライズ |

実装単位の案は `src/heretic/modifiers/soma.py`、独立した SOM 候補生成処理、低 rank operator の計算処理、対応する設定と検証。上流は #446 で既存の可視化・研究機能を削除する方針なので、可視化を必須依存にせず解析用 artifact を出す設計が合う。なお #446 は同時に複数 modifier を使う構成をまだ拒否する。SOMA と ARA を併記するだけで合成できるわけではない。[初期化・制約](https://github.com/p-e-w/heretic/blob/ce8b05c77b82f0b350e137d7b783a2eb5e1059db/src/heretic/modifier.py#L108)

探索は最初から「全層 × 全方向 × component ごとの独立 min/max」にしない。次の順序が検証しやすい。

1. 既存の単方向 baseline と固定 source layer の SOM 候補を用意する。SOM の格子・学習率等は固定し、毎 trial の再学習を避ける。
2. 参考実装に対応する sequential を作り、候補の順序を保って探索する。初期段階では k と component を固定し、方向列の選択を評価する。
3. 同じ候補に additive を適用し、共有の層別強度と少数の混合係数で比較する。非負係数の和を 1 に固定すれば、候補が増えただけで総強度が膨らむ問題を抑えられる。ただし逐次式とは別の手法であり、性能が上がる保証はない。
4. 有望な候補に対して FULL、good 平均への直交化、component の選択を一つずつ追加する。MLP を無効化できる現 master の選択肢も残す。
5. 層別 SOM、層間補間、候補ごとの自由な強度は、その追加コストに見合う改善が見えた場合に拡張する。

Optuna では、trial ごとに候補の定義・分布を変える設計を避ける。sequential は順序の違う方向列を別候補として扱い、additive は順序に依存しないため重複評価をまとめられる。候補 ID に意味のない大小関係を与えない設計、固定探索空間、同じ条件の重複評価 cache を検討する。1 本の平均方向に戻す baseline はそのまま残す。有限反復の 1-neuron SOM が、標準設定のまま厳密に平均に一致するとは仮定しない。

参考実装も、そのまま流用するより小さく取り込む方がよい。`som_generate_directions.py` は実データで勝者頻度を数えるが、勝者になった neuron だけを頻度順に出力するため、候補数が常に格子数になるとは限らず、配列 ID も固定格子 ID ではない。README の Llama3 の方向列には同一 ID の重複がある一方、探索コードと論文は重複を排除する設計である。方向 tensor と ID の対応・順序を保存し、再現する artifact を明示する。[候補の保存](https://github.com/pralab/som-refusal-directions/blob/d244c7d282ac65a1520bef0d418615ef148108af/som_generate_directions.py#L71)、[README](https://github.com/pralab/som-refusal-directions/tree/d244c7d282ac65a1520bef0d418615ef148108af)

依存は MiniSom と必要な数値計算だけに限定できる。参考実装の可視化・モデル wrapper・古い Transformers 一式の導入は不要。SOM の topology、sigma、learning-rate / sigma decay、初期化、seed と MiniSom 版を明示し、現在の Heretic の依存を不用意に古くしない。

メモリ面では、現在の `get_residuals_mean()` が平均だけを逐次集計している点を維持したい。good 側は平均だけで足りる。bad 側は選んだ層の表現だけ CPU に保存すれば、全層の大きな tensor を保持する必要を減らせる。source layer を選び終えてからその層だけを蓄積するか、候補層を限定する。複数層を必要とする場合は CPU / disk cache のキーに model revision・template・prefix・データ hash・加工設定を含める。[現在の平均集計](/mnt/ssd1/heretic/src/heretic/model.py:760)

**実モデルでの採用判断は、同条件の Pareto 比較で行う。** #196 の作者は Gemma-3-12B で拒否数 3/100 のまま KL 0.16 → 0.08 などの結果を報告した。一方、メンテナは後に ARA が GPT-OSS-20B で同じ拒否数・約半分の KL を得たと報告している。いずれも投稿者の実験報告であり、今回の追試結果でも、現 master に対する優越性の証明でもない。[PoC の報告](https://github.com/p-e-w/heretic/pull/196)、[ARA の比較報告](https://github.com/p-e-w/heretic/pull/196#issuecomment-4003490899)

必要な比較条件は以下。

- 同じ model revision、prompt / prefix、量子化、生成長、SOM 候補、評価データを固定する。
- 候補学習用、Optuna 選択用、最終 test 用を分離する。最終 test で追加探索しない。
- 現行単方向、旧 PoC 相当、修正版 additive、sequential を比較し、上流統合を目指すなら #446 の ARA も含める。
- 同じ trial 数と同じ壁時計予算の双方で比較し、複数 seed で変動を確認する。
- キーワード拒否率、judge 評価、初回 token の KL、一般能力の benchmark、時間、ピークメモリを測る。日本語用途なら日本語の別評価も含める。
- 方向数、rank、FULL、good 平均への直交化を分けて ablation study を行う。

Heretic の KL は最初の出力 token の分布差であり、能力保持全体を保証しない。また拒否キーワードがないことは、要求に適切に応じたことを意味しない。PR 内には Qwen3.5 でキーワード判定と judge 判定が大きく食い違った報告もある。小さな拒否数差だけで優劣を断定しない。[現在の KL scorer](/mnt/ssd1/heretic/src/heretic/scorers/kl_divergence.py:43)、[judge 比較の報告](https://github.com/p-e-w/heretic/pull/196#issuecomment-4022805436)

工学的な受入条件として、単方向経路の互換性、ゼロ方向・重複・空データ・無効な k の扱い、試行 A→B→A の復元、CPU / dtype / device_map、adapter 保存と merge 後の再ロードを確認する。非量子化モデルから始め、BNB の場合は量子化した基底に対する評価と、非量子化基底への merge の差も測る。MoE の fused expert や hybrid attention は、既存 module discovery の対象範囲を超えて自動的に対応できるとはしない。

再現情報についても #446 は `reproduce.json` を version 3 から 4 へ変更する。SOM artifact の再生成か保存 tensor の利用かを定め、旧 #196 の checkpoint をそのまま新形式として読み込まない。外部 plugin の試作と、Heretic の組込み plugin として再現性を提供する段階も分ける。[#446 の変更](https://github.com/p-e-w/heretic/pull/446/files)

以前の CLE / SteerEdit の調査との関係では、SOMA はまず方向候補の抽出と operator として独立させると再利用しやすい。今回の rank k 以下という結果は、指定した方向の線形な重み更新についてのもの。CLE の入力依存介入を重みに近似する回帰問題が不要になるわけではない。また、#446 の現在の API で SOMA + ARA + CLE を単純に積み重ねられるわけでもない。

実装を始める場合の推奨成果物は、**#446 の API に沿う独立 SOMA modifier、修正済み候補選択、none/pre の正確な低 rank 更新、FULL の近似誤差計測、比較用の実験設定**。まずこれで既存 PoC の工学的問題を解消し、実モデルで性能と費用の改善が確認できた段階で既定値・対応モデルを決める。
