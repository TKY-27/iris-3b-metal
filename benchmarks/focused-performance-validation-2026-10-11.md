# Focused performance validation — 2026-10-11

> Post-release audit follow-up: the unmerged fresh-shadow implementation,
> `--shadow-json`, and `--cfg-interval` experiment flag were removed from
> maintained scripts. The original diagnostic source and incoming diff remain
> under `.scratch/post-release-audit/`; the evidence below records the original
> run, and its shadow commands require that archived code. Batch fixture
> support, prompt profiling, and solver branch-order regression coverage remain.


今回の候補から、品質と実測性能の両方を満たす高速化は採用できなかった。ランタイムの `src` は作業開始時と全ファイルSHA-256一致。残す変更は任意の計測機能、fresh計算のみのshadow診断、回帰テスト、本記録。近似のランタイムAPI・既定設定は追加していない。公開・push・PR・リリース・コミットは行っていない。

## 基準と再現条件

- HEAD: `0beea7be41b9b42a592c08f7411d5d38930c4d5f`。開始時の非ignored dirty/untrackedはゼロ。レビュー基準と同一。Turboのチェックポイント・実装・係数は導入していない。既存のignored作業は保護。
- Apple M5（10 GPU cores）、24 GiB、macOS 27.0.1 / 26A434、Python 3.14.8、PyTorch 2.14.1 / `5c4886908584029761b579af026dcfb627c84070`、Transformers 5.19.0。AC給電、lowpowermode=0。
- `HF_HUB_OFFLINE=1`、既存の `.venv`。MPS/PYTORCH環境変数の上書き、CPU fallback、fast mathの追加なし。既存mixed precision（FP16 trunk linear、BF16 text adapter/PiT/Qwen、FP32 norm・残差）、FP16 DiT attention。CFG=3、Flow DPM-Solver++ order=2、shift=4、velocity prediction、interval=(0,1)、1000 training timesteps。試験は1024×1024、10/20評価。steps・精度・solver・出力変換の既定値は変更なし。
- DiT revision `7445443349bc9abe3c96f01ff793e2098ca012b3`、実測SHA-256 `a90e1f4c197bac8491f82078def68884b6a3527ea7fc97a693e3e367372aa80f`（11,950,112,680 bytes）。Qwen revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`。重みの複製・変換・ダウンロードなし。
- 実際の開始時ツリーを [runnable baseline](../.scratch/focused-20261011/baseline) に保存し、全trackedファイルを [baseline-identity.json](../.scratch/focused-20261011/baseline-identity.json) に記録。`PYTHONPATH=.scratch/focused-20261011/baseline/src` で起動可能。checkpointは元の場所を参照する。
- LUT・近似比較では既存cat-window/sign-bread raw conditioning（padding300、mask・branch順序保持）と保存済みの実ノイズを使用。seedだけで比較していない。最終hashは [final-identities.json](../.scratch/focused-20261011/final-identities.json)。
- 同時にこちらからGPU計測ジョブを複数起動していない。無関係なプロセスを停止していない。一方、背景GPU負荷を十分低く固定できず、後半の基準計測開始時は32.0% / 46.2%。値は利用率で正規化していない。OS file cacheはunpurgedで、filesystem coldは未確立。制御されたidle条件での性能採用証拠は得られていない。

## 基準の観測時間

各step数は別のfresh process。benchmarkのrepeats=3は初回1回＋同一prompt warm2回。単位秒。別の `--profile --profile-prompt` リクエストは診断専用で、無計測リクエストの値と混ぜていない。

| 区間 | 10 steps | 20 steps |
|---|---:|---:|
| Imports | 0.682023 | 0.692849 |
| Config初期化 | 0.035454 | 0.019687 |
| Tokenizer/text setup | 1.439180 | 1.361827 |
| 初回raw Qwen encode | 3.161973 | 2.464748 |
| DiT load | 7.646744 | 6.677241 |
| 初回generation（準備後） | 45.709674 | 96.296555 |
| Imports＋初回request準備込み | 58.677146 | 107.521623 |
| 同一prompt warm median | 45.319481 | 96.758631 |
| Warm engine uncached total | 56.051246 | 104.052498 |
| 診断uncached raw encode | 5.328992 | 6.706898 |
| 診断request→初回host submission | 5.721157 | 7.146287 |
| 診断request→初回coarse completion | 10.083247 | 15.122438 |
| 診断初回complete evaluation | 4.361057 | 7.975156 |
| 診断steady evaluation median | 4.470492 | 4.582757 |
| 診断pure denoising | 44.679859 | 95.862565 |
| 診断image conversion | 0.005027 | 0.007576 |

Raw warm: 10=[44.567041, 46.071920], 20=[97.977544, 95.539718]。診断adapterの2回は10=[0.294045, 0.088678], 20=[0.345537, 0.087776]。wrapperがadapted cacheを無効化するためmissの診断。end eventは各1回だけ読む。per-layer同期は追加していない。
Raw: [baseline-10.json](../.scratch/focused-20261011/baseline-10.json), [baseline-20.json](../.scratch/focused-20261011/baseline-20.json)。計測レシピ: [recipe.json](../.scratch/focused-20261011/recipe.json)。

## Exact BF16 GELU lookup — 数値PASS、性能採用FAIL、撤回

同じGPU・同じcompile_shader設定・round_bf16・literal c10::metal::erf式で65,536-entryの通常device BF16 buffer（128 KiB）を生成した。BF16丸め後FP32の上位16bitで索引、nonfiniteは元の式へfallback。FC順序・dispatch geometryは維持。MetalIrisがlazy所有し、CPU/unsupported dtype参照経路は維持。

- 65,280 finiteパターン全てbitwise一致（±0・subnormal含む）、全65,536の出力bit patternも一致。実kernelの±inf/NaN入力一致。
- 全4 PiT blocks × early/middle/late × 10/20 schedules＝24実checkpoint pixel比較一致。6固定状態velocity一致。両scheduleのcomplete rollout最終FP32状態一致、6 paired generationのuint8画像一致。全評価の状態アーカイブは保持していない。
- 24×24 / 24×16 tiny MPS比較一致、model削除でlazy table解放。B=n条件行の非zero storage offset sliceをn=1/2でcopyと比較して一致。実1024 B1対B2 conditional rowsも固定状態bitwise一致。
- fresh-process compile＋table初回は0.118783 s、既存libraryからのtableは0.003206 sの初回観測。別の保存済みpilot runでは0.004248 s。cold requestの純増分を対にして測定した値ではない。
- 3 AB/BA warm pairs/step count。正のdeltaはcontrol−lookup＝改善。

| Steps | Control raw seconds | Lookup raw seconds | Paired delta seconds | Median delta | 小標本bootstrap median interval |
|---|---|---|---|---:|---|
| 10 | 39.369460, 41.411534, 43.007433 | 39.586481, 42.258010, 42.297656 | -0.217022, -0.846476, 0.709777 | -0.217022 | [-0.846476, 0.709777] |
| 20 | 84.262341, 81.356704, 83.331744 | 84.856626, 87.338100, 81.547483 | -0.594285, -5.981395, 1.784261 | -0.594285 | [-5.981395, 1.784261] |

同一controlのrangeは10=3.638 s /20=2.906 s。これは反復controlのspreadであり、隣接randomized A/A対の推定ではない。3pair bootstrapは探索的で小標本の限界が大きい。どちらも改善の符号が不安定で、中央値は悪化。noiseを超える生成利益なしとしてfinalistにせず、5pair採用試験へ進めなかった。
固定状態forward 3pairsのcontrol median C1=1.838373 s、C2=3.747381 s。lookup C1=1.826869 s、C2=3.712573 s。C2-C1=1.909007 sを基にCFG差分の粗い最大節約を3hit×=5.727 s /8hit×=15.272 sと見積もった（copy等を含まない）。2倍速度と仮定していない。kernel単体は各block約5.5–6 msで、3pairsの差は混在。
Raw: [lut.json](../.scratch/focused-20261011/lut.json), [lut-summary.json](../.scratch/focused-20261011/lut-summary.json), [lut-extra.json](../.scratch/focused-20261011/lut-extra.json), [kernel-bench.json](../.scratch/focused-20261011/kernel-bench.json)。試験ソースのみ [lut-source](../.scratch/focused-20261011/lut-source) と [lut_pilot.py](../.scratch/focused-20261011/lut_pilot.py) に保存。`model.py` / `kernels.py` の実験変更は元のツリーへ復元し、runtime toggleは残していない。

## Warm uncached prompt処理 — prefix K/Vはdefer

固定prefix=34 tokens、suffix=5。提案KV量は4,874,240 bytes（約4.65 MiB）。DiT residentで3つの未cache短captionを測定した。

| Prompt | encode diagnostic | layer compute sum | weight-copy window sum |
|---|---:|---:|---:|
| ceramic cup | 2.920863 | 0.192277 | 2.554851 |
| stained glass greenhouse | 1.585584 | 0.163989 | 1.365842 |
| glass terrarium | 2.088927 | 0.163122 | 1.834370 |

copy windowはhost weight accessの隙間を含み、純粋なblit時間ではない。それでもlayer計算全体は約0.16–0.19 sで、streaming/copy/lifetime側が支配的。指定のdefer条件に従いprefix-KVを実装していない。readers/sourcesは各request後0。Raw: [text-profile.json](../.scratch/focused-20261011/text-profile.json)。

## CFG差分・既存CFG interval — 実画像のcritical regressionで棄却

CFG差分はrequest-local closureとして実装し、FP32 owned d_anchor、original full result直接return、fresh conditional hit、batch sliceのみ、最初2/最後2 full、連続hit禁止、unguided/error/cancelで無効化した。n=2 toy、短schedule、constant/changing差分、再利用出力alias、mode transition、forced-full/unsupported recipe、cancel後の次requestを試験。NFEは10/20のまま。

| Method | NFE 10 / 20 | Full guided 10 / 20 | Conditional-only 10 / 20 | Reused middle |
|---|---|---|---|---|
| Original | 10 / 20 | 10 / 20 | 0 / 0 | 0 |
| CFG difference pilot | 10 / 20 | 7 / 12 | 3 / 8 | 0 |
| CFG interval (0,0.95) | 10 / 20 | 8 / 16 | 2 / 4 | 0 |

実time_gridの先頭は0.9997498123592695、strict predicateで確認。(0,0.9)のguided countは6/13（unit/grid確認のみ、画像は未実施）。solver historyは変更・restartしていない。両手法の組合せなし。

開発最初の明示的typography prompt/sign-bread、2 seeds、両step数を実施。seed1/10でbaselineの「FRESH BREAD」に対しCFG差分は「BIREAD」と読める余分なI様glyph。interval095はseed1の10/20両方で右側に重複看板・不正な文字が発生。seed0やCFG差分seed1/20では同じcritical failureは見えず、prompt/seed/step依存がある。実装担当者が直接paired画像を確認したunblinded reviewであり、独立blind評価ではない。
同じ保存noise/conditioningでseed1/10をfresh process再生成し、各手法のPNG bytesがそれぞれ初回と一致して破綻再現。[reproduced-critical.json](../.scratch/focused-20261011/reproduced-critical.json) と [visual-review.json](../.scratch/focused-20261011/visual-review.json)。画像: [baseline](../.scratch/focused-20261011/sign-bread-seed1-10-full.png), [CFG difference](../.scratch/focused-20261011/sign-bread-seed1-10-cfg-difference.png), [interval095](../.scratch/focused-20261011/sign-bread-seed1-10-interval095.png)。
critical regressionによる早期棄却なので残り7開発prompt、24×3×2 holdout、finalist性能5pairs、non-square近似画像、(0,0.9)画像、組合せは未実施。等価性・quality non-degradationは主張しない。

以下はfixed-order品質screenの純denoise/生成（conditioning準備後、画像変換まで）であり、paired warm acceptanceではない。encoding込みE2Eの候補速度改善は未確立。

| Seed | Steps | Full denoise / total | CFG diff denoise / total | Interval095 denoise / total |
|---|---|---|---|
| 0 | 10 | 36.092225 / 36.097282 | 31.689784 / 31.693616 | 33.843026 / 33.846849 |
| 0 | 20 | 75.744038 / 75.747772 | 61.754491 / 61.758215 | 69.746657 / 69.750573 |
| 1 | 10 | 38.858167 / 38.861983 | 34.265254 / 34.269156 | 36.811414 / 36.815260 |
| 1 | 20 | 82.250299 / 82.254317 | 69.946093 / 69.949944 | 82.630742 / 82.635029 |

CFG copy/reconstructionの単独costは未分離。Raw [approx.json](../.scratch/focused-20261011/approx.json)、機構検証 [cfg-toys.json](../.scratch/focused-20261011/cfg-toys.json)。不採用CFG pilotは [cfg_pilot.py](../.scratch/focused-20261011/cfg_pilot.py) にローカル再現資料として保存するのみで、tracked runtimeや既存GUI/HTTPには接続していない。interval flagは既存solverのbenchmark実験用だけで既定値不変。

## Fresh shadowとmiddle reuse screening

8-dual/16-single topologyでprefix 0–1、middle 2–21、suffix 22–23を観測。original separate contiguous image/textを維持。I/P/Rはowned、suffixのin-place mutation前にRを計測。全forwardとPiTはfresh、original resultをそのまま返す。各requestで例外・cancel・終了時にhook/anchorを解除し、reportを残してもengineを保持しない。cache-offにはsnapshot/reduction/readbackなし。

| Steps | Full / reused | Final FP32 equality | Would hit (.01/.03/.05) | Eligible prefix max metric min–max | Actual middle image reconstruction relative error min–max |
|---|---|---|---|---|---|
| 10 | 10 / 0 | bitwise PASS | 0 / 0 / 0 | 0.114186, 0.221720 | 0.169892, 0.398941 |
| 20 | 20 / 0 | bitwise PASS | 0 / 0 / 0 | 0.057770, 0.231706 | 0.056185, 0.535975 |

3つの小閾値でeligible hitなし。shadow誤差は品質boundではない。大きい閾値へ広げて採用しない。stale dual textだけを示す結果でもないため代替layoutも実装せず、middle runtime cache・reduction融合はdefer。
coarse median prefix/middle/reconstruction/reduction: 10=0.342521 /3.451703 /0.002765 /0.020286 s、20=0.335960 /3.505069 /0.002204 /0.019892 s。host reduction+readback欄は既にqueueされたmodel completionの待ちを含み、純readback/decision時間ではない。動的decision実行はゼロ。
Raw [shadow-10.json](../.scratch/focused-20261011/shadow-10.json), [shadow-20.json](../.scratch/focused-20261011/shadow-20.json), [shadow-summary.json](../.scratch/focused-20261011/shadow-summary.json)。完了progressはそれぞれ1..10 /1..20、shadow operations後にqueuedを1回。
診断のfrozen-history CFG one-step drift式はレビューでCFGからCFG−1へ訂正した（fresh c、u_hat=c−dのため）。solverのfirst/second/terminal updateに対する独立toyで検証済み。保存済みnative rawの古いdrift欄は上書きせず、`shadow-corrected-10/20.json` に2/3倍の解析補正とprovenanceを保存した。この補正値はGPU再測定ではない。他の指標、would-hit、画像・最終一致は影響を受けない。

## Memory、runtime trace、検証

初回基準のphysical stageピークはtext≈1.526 GiB、DiT load≈6.53 GiB、generation≈10.02 GiB。MPS current=6,105,174,784 bytes（5.686 GiB）、driver=9,801,203,712（9.128 GiB）。LUT試験のphysical peak≈10.25 GiB、current=6,191,354,880、driver=9,851,551,744で全pair一定。両modeにtableがresidentなので、128 KiB以外のcandidate-only memory因果比較ではない。
shadow off/onのphase peakはともに約9.85–9.88 GiB（既存allocator pool共有、fresh isolated peak差ではない）。終了時currentは6,128,243,456→6,321,181,440 bytes（約184 MiB増）、driverは9,817,980,928で一定。隣接残差のみ保持しGPU trajectoryは保持しない。transient snapshot/workspaceの最大量をpersistent bytesから代用しない。
global16 KiB/pageのPageouts/Swapouts delta: 基準10=338/392585、20=398/390847。swap-used変化は+1820.81 /−2818.69 MiB。LUT10=36/0、20=10/30559。shadow10 on=0/8161、20 on=0/0、pressure level=1。基準でも大量のglobal swap activityがあり、候補への帰属は未確立。swap-usedの減少をswap I/Oなしとは扱わない。候補のswap-free/独立memory-pressure acceptanceは満たしたと主張しない。
実checkpoint B2 forwardの別診断は4.410105 s。coarse event合計linear=2.880077、SDPA=1.019007、4 pixel MLP=0.027312 s。主なtrunk costはGEMM/SDPA。normイベント1個はunavailable。shader occupancyや個別hardware counter由来の割合ではない。
Metal System Traceの実dispatchに `prefill_attention_mpp_a10_float16_bq64_bk32_bd128_wm4_wn1_hm0_dc0_maskfloat16` およびBF16 MPPを確認。ShaderTimeline 4574 rows /2280 named dispatches、GPU interval15236 rows。shader-profiler interval0 rowsなので詳細counterは未取得。[trace-summary.json](../.scratch/focused-20261011/trace-summary.json)、[metal.trace](../.scratch/focused-20261011/metal.trace)。

- PASS: Metal engine/kernels/text/startup/pipeline/CLI/server/loading関連＋flow/Qwen/overflow/adapterの152 tests（37.68 s、4既存warning）。n2 branch order、非contiguous hazard、alias所有権、nonfinite、unsupported topology、cancel後request、error cleanup、unload、event read-onceを含む。
- PASS: `denoise_fixture.py` 実CLIでn=2、24×16、tiny topology、2stepの通常/shadow最終bitwise一致。steps=0 / NaN intervalをparserが拒否。temporary toy checkpointは検証後に削除。
- PASS: `ruff check .`、`ruff format --check .`（118 files）、`git diff --check`。runtime source hash一致。
- 未実施: GUI実機操作再検証、全training suite、独立blind holdout、制御されたidle背景での採用計測。GUI/Python/HTTP production sourceは変更なし。テスト通過をこれらの実動作証拠へ読み替えていない。

## 残した変更と再実行

- `scripts/benchmark.py`: 任意 `--profile-prompt`、raw encode wallとadapter coarse events。独立診断で、通常request設定は不変。
- `scripts/denoise_fixture.py`: n=2 fixture対応、`--shadow-json`（別full trajectoryのbitwise最終比較）、既存solverへの`--cfg-interval`実験用指定、入力validation。
- `scripts/reuse_diagnostics.py`: 3閾値以内のfresh shadow、per-branch image/text/spatial summariesとbounded隣接residual。近似計算やGUI/HTTPの新機能なし。
- 既存engine/flow testsに小さな機構・回帰ケースを追加。

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark.py --checkpoint iris-3b --size 1024x1024 --steps 10 --repeats 3 --profile --profile-prompt "A distinct uncached prompt" --out .scratch/baseline-new.json
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python scripts/denoise_fixture.py YOUR_SAVED_NOISE_FIXTURE.npz --checkpoint iris-3b --steps 10 --repeats 3 --out .scratch/fixture-new.npz --shadow-json .scratch/shadow-new.json
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_metal_*.py tests/test_flow.py tests/test_qwen3_vl_text_encoder.py tests/test_caption_overflow.py tests/test_text_adapter.py -q
```

fixtureには比較対象と同一の実noiseを格納する。今回native fullモデルの測定レシピは `.scratch/focused-20261011/` 内の各runner/JSONに保存。failed runtime togglesは取り除き、再現資料は自分のignoredディレクトリにのみ残した。今回の境界内で追加実験・公開作業を行わず終了する。
