# SOM モジュールの使い方

この fork では `modifier = "som"` で SOM、`modifier = "abliteration"` で従来方式を選びます。既定値は従来方式です。本家のプラグイン PR の取り込みは不要です。

リポジトリで次を実行します。

```bash
uv sync --extra som
uv run --extra som heretic --model MODEL_ID --modifier som
```

設定を調整する場合は `config.som.toml` を参考に、使用中の `config.toml` に設定を追加します。既存のプロンプト・評価データ等の設定はそのまま利用できます。

```toml
modifier = "som"
som_direction_scope = "auto"
som_grid_size = 4
som_directions = 4
som_iterations = 10000
som_learning_rate = 0.01
som_sigma = 0.5
som_max_weight = 4.0
row_normalization = "full"
full_normalization_lora_rank = 8
```

初期処理では、good 表現の平均と bad 表現を採取して SOM を学習します。`som_direction_scope` は次の3種類です。

- `"auto"`（既定値）: 全 transformer 層で SOM を一度学習し、Optuna が trial ごとに global / per layer を選びます。
- `"global"`: 中盤〜後半の層で SOM を学習し、Optuna が整数の source layer を選びます。その層の複数方向を各対象層の出力行列へ適用します。
- `"per layer"`: 全 transformer 層で SOM を学習し、各層の方向をその層の出力行列へ適用します。層間補間はしません。

層ごとの適用強度は、いずれの方式でも従来同様に探索します。CLI では `--som-direction-scope "per layer"` のように指定できます。

初期処理を短くしたい場合は、`som_direction_scope = "global"` と、モデルの層数に合う `som_source_layer = 15` のような設定で source layer を固定できます。添字は 0 始まりで、embedding slot は含みません。`auto` で source layer を固定しても global の選択だけに作用し、全層の学習は行います。`per layer` と source layer 固定の同時指定はエラーになります。学習は各 source layer につき一度だけで、trial ごとには再学習しません。

格子は既定で `som_grid_size = 4` の 4×4 です。長方形を使う場合は `som_grid_shape = [3, 5]` のように指定すると `som_grid_size` より優先されます。

**#196 からの主な修正:** 実サンプルの BMU 割当数で候補を順位付けし、ゼロ・重複方向を除外します。不足方向は複製せず実際の候補数で動作します。独立に学習した SOM 同士の補間は行いません。方向の混合比は component 間で共有し、使われない方向別 min/max パラメータを持ちません。

`som_directions` は採用方向数の上限です。bad サンプルが少ない、同じ表現ばかり、good 平均への直交化で方向が消える、といった場合は実際の本数が減ります。per layer では方向が得られない層をスキップして通知し、その層の重みを変更しません。利用可能な方向が全 source layer で 0 本なら、データを確認できるよう明示的に停止します。

per layer の混合係数は、各層内での実サンプル支持数の順位に対する重みとして共有します。同順位が同じ概念だとは仮定せず、方向自体は各層のものを使用します。層ごとに実際の方向数で混合比を正規化するため、1本しかない層はその1本に重み1を与えます。各層・各 component・各方向の混合係数をすべて独立に探索する実装ではありません。

混合比の合計は 1 です。`som_max_weight` は方向ごとの上限ではなく、**方向全体に掛ける強度の探索上限**です。#196 のように本数を増やすだけで合計強度が増える挙動を避けつつ、強い更新も探索できるよう初期値を 4.0 にしています。この値が全モデルで最適と確認済みという意味ではありません。

`row_normalization = "none"` / `"pre"` では多方向の加算更新を直接 LoRA の A/B に分解し、rank 1 へ圧縮しません。`"full"` では行ノルム復元後の差分を SVD 近似します。SOM の adapter rank は `max(実際の最大方向数, full_normalization_lora_rank)` です。FULL の近似誤差が気になる場合は後者を増やします。LoRA の保存容量と計算量は増えます。

このモジュールは #196 を改良した**加算方式**です。論文の参考実装が採用する順序付き逐次更新の完全再現ではありません。対象行列は Heretic が元から扱う attention / MLP 出力行列で、embedding を新たに編集することはありません。`print_residual_geometry` / `plot_residuals` は SOM でも利用できます。これらを指定した場合に限り解析用の全 good/bad 残差を保持するため、通常実行よりメモリを消費します。解析内容は従来の平均差方向・残差分布であり、SOM ニューロンごとの可視化ではありません。

**保存と再開:** 標準方式は従来の checkpoint 名、SOM は `--som.jsonl` の名前で保存するため、同じモデルでも両方式を混ぜません。再開時は保存された設定が優先されます。SOM の格子・本数・source layer 等を変えて新しく試す場合は `--checkpoint-action restart`、または別の `--study-checkpoint-dir` を使います。restart はその方式の既存 checkpoint を削除する従来の操作です。

adapter 保存と merged model 保存は従来のメニュー・オプションから使えます。マージ後に別 trial を選ぶ場合も、正しい rank で adapter を再作成します。試行には scope、source layer または層別の neuron ID・混合比、各 component の強度を保存します。再開時は同じデータと seed で候補を再計算します。scope 導入前の SOM checkpoint / 再現ファイルは global として読み戻し、自動的に auto へ切り替えません。ハードウェアをまたぐ完全一致や、大規模な研究 artifact 管理を目的とした構成ではありません。

従来の version 3 の再現ファイルと checkpoint は従来方式として読み込みます。新しい再現ファイルは、この fork の内部モジュール形式である version 4 を出力します。本家 PR #446 の version 4 との互換性を示すものではありません。

**新しい方式を追加する場合:** `src/heretic/modifier.py` の `Modifier` を継承し、`init(good_prompts, bad_prompts)`、`suggest_parameters(trial)`、`modify_model(parameters)` を実装します。パラメータは JSON 化できる辞書を返します。通常の LoRA 方式は既定の `reset_model()` を使い、独自の重み編集や hook を持つ方式はそれも実装します。`create_modifier()` のレジストリと `Settings.modifier` の選択肢へ追加すれば、main の探索・選択・保存経路を再利用できます。

モデル API の `apply_lora(rank)` は初期化時に使います。既存 adapter をマージせず外して rank を変更するため、変更済みモデルへ後から呼んで変更を保持する API ではありません。方向抽出は `get_residuals()` / `get_residuals_mean()`、加算型の更新は `abliterate_multiple()` を利用できます。

検証はネットワークなしの CPU で実施できます。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  uv run --extra som python -m unittest discover -s tests -p 'test_*.py' -v
```

小型ランダム Llama と PEFT を使って、数式との一致、候補の選択、試行のリセット、adapter 保存、merge 後の再ロード、CLI の保存・再開を確認します。これらは実モデルの拒否率や能力保持の改善を測るベンチマークではありません。大規模モデル、BNB 4bit、複数 GPU での性能は別途確認が必要です。
