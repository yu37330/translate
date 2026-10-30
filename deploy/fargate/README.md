# ECS Fargate版（CPU Whisper）

本ディレクトリはリポジトリ直下の `app.py` / `templates/index.html` をコピーして、社内ネットワーク向けPoCとして独立したものです。**元のアプリは変更していません。**

## できること

- ブラウザーのマイク、または共有画面のシステム音声を取得し、16 kHz / 16-bit PCM（0.5秒ごと）をWebSocketで送信
- NLB → FargateのFastAPI `/ws/audio` が受信し、**同じPythonプロセスのfaster-whisper（CPU）** で認識
- 同じWebSocket経由で文字起こしを返却。Whisper用の別APIやGPUは不要
- 0.5秒の**送信チャンクは分割輸送にのみ使用**し、サーバーは4〜12秒の連続音声を蓄積してWhisperへ渡す。発話終了（約0.8秒の無音）で確定。開始前0.3秒も先読みバッファとして保持
- ブラウザー側の音量しきい値による破棄を廃止し、サーバーの緩い音量検出＋faster-whisper内蔵Silero VADで処理。Stop時には端数PCMも送信して未確定発話を確定
- `ASR_VOCAB_PROMPT`で専門用語辞書のヒントを指定可能。直近確定発話を短く次の推論に渡す（設定で無効化可能）
- Whisperの暫定結果は文章全体を置換、確定時に一度だけ履歴へ保存。Bedrock翻訳は別の非同期処理で実行し、音声認識キューを塞がない
- モデルはイメージのビルド時に取得して同梱。実行時はモデルの外部ダウンロード不要

**重要:** Teams標準の文字起こしを取得するものではありません。PCで共有できるTeams音声をブラウザー経由で再認識します。システム音の取得可否はOS・Teams（デスクトップ/ブラウザー）・ブラウザーの仕様に依存します。また今のUIではマイク/システム音は二者択一であり、会議全員の発言取得は保証しません。

## 音声認識の精度改善と調整パラメータ

初期設定は**4秒単位で暫定字幕**を更新し、約0.8秒の無音または**最大12秒**で確定します。送信チャンクを0.5秒に短縮し、発話開始前0.3秒を保存して語頭を切りにくくしました。CPU負荷と遅延のトレードオフがあるため、4 vCPU Fargateの実測後に調整してください。

| 環境変数 | デフォルト | 用途 |
| --- | --- | --- |
| `ASR_BEAM_SIZE` | `3` | 1〜8で調整。上げると精度が改善する場合があるがCPU負荷増 |
| `ASR_SPEECH_THRESHOLD` | `0.0006` | サーバーの緩い音量ゲート。小声を落とす場合は小さくする |
| `ASR_END_SILENCE_SEC` | `0.8` | 発話確定に必要な無音の長さ。短くすると文の途中切れリスク |
| `ASR_PREROLL_SEC` | `0.3` | 発話開始前の音声を最大0.3秒保持し語頭の欠落を防ぐ |
| `ASR_HOTWORDS` | 空欄 | faster-whisper `hotwords` で専門用語候補を追加（音声で正しく話した用語の評価に使う） |
| `ASR_VOCAB_PROMPT` | 空欄 | 工場・専門用語の例（BOM、ECN、PLM等）。機密情報ならECS環境変数の管理ポリシーに従う |
| `ASR_PREVIOUS_CONTEXT_CHARS` | `80` | 直近確定結果から文脈として渡す最大文字数。誤補完が増える場合は`0`にする |
| `ASR_MIN_INTERIM_SEC` | `4` | 最初の暫定表示までの蓄積秒数 |
| `ASR_INTERIM_INTERVAL_SEC` | `4` | 途中結果の再認識間隔 |
| `ASR_MAX_UTTERANCE_SEC` | `12` | 1回の最大認識長。長い連続発話はここで強制確定 |

**注意事項:** これは精度向上を狙った実装ですが、実際のCER/WERや遅延改善は未計測です。長いチャンクは推論コストが増えます。Silero VADも含めて小声・雑音・無音時の誤認識を確認してください。専門用語プロンプトは誤補完や幻覚を招く場合があるため、原音との突き合わせが必要です。ユーザー別メモリ・話者識別・会議録音保存は追加していません。

## Vexaのノウハウから取り入れた字幕安定化（第三弾）

GitHub [Vexa](https://github.com/Vexa-ai/vexa) のmixed-pipeline / speaker-streamsの設計を参考に、依存関係・ボット・話者分離は**移植せず**、Fargate単一コンテナのPythonで以下を再実装しました。

- **LocalAgreement-2（表示専用）**：同じ発話について連続する2回のWhisper仮説から日本語も使える文字単位の共通接頭辞を抽出し、UIで下線表示。**確定保存は従来どおり発話終了時だけ**。日本語のスペース区切りに依存せず、後続文字を暫定扱いにする。早期確定（真のVexa LocalAgreement）は実装していない
- **正確な音声時間**：ブラウザー音声を16kHzサンプル単位で積算し、各発話に`audio_start_ms`/`audio_end_ms`と`utterance_id`を付与。暫定字幕と最終字幕の音声区間を共有し、同じ`utterance_id`の最終字幕重複保存を防ぐ。これは**音声区間の時刻**であって、Whisper単語タイムスタンプや話者分離ではない
- **無音でプロンプトリセット**：`ASR_CONTEXT_RESET_SEC=3.0`（既定）以上の連続無音をPCM音声時間から計測し、前の確定発話を次のWhisperプロンプトに引き継がない。専門用語の静的ヒントは残す
- **慎重な幻覚抑制**：Whisperの`no_speech_prob >= 0.95` と `avg_logprob <= -1.1` が**同時に成立するセグメントのみ**除外。反復フレーズはテキストをログに出さず件数観察だけで、削除しない。「はい」「そうですね」など正当な短い応答の一律削除はしない
- **リプレイ評価**：`replay_asr.py`でローカルの承認済みPCM WAVを0.5秒刻みで通し、確定文字起こしCER、発話数、暫定一致イベント、RTFを集計。**音声・文字起こしの本文は出力しない**。AWSやHugging Faceへの送信もない。ネットワーク/ブラウザーの遅延は含まれない

| 環境変数 | 既定 | 用途 |
| --- | --- | --- |
| `ASR_CONTEXT_RESET_SEC` | `3.0` | 前の確定文章の音声文脈をクリアする無音秒数 |

**改善効果は未測定**です。Vexa独自の話者識別、早期確定、Pyannote推論、単語アラインメントはまだ導入していません。文字ベースのLocalAgreementは表示上の参考情報であり、誤った字幕を確定済みと扱わないようにしています。

## 実機試験前の精度・信頼性対策（第四弾）

- **0.5秒PCM内の発話境界を100ms刻みで判定**：1パケットの前半が無音・後半が次の話者の発話だった場合、次の発話を破棄せず、2件の音声区間として継続。音声サンプル時計と確定字幕IDの整合も維持
- **セッションIDの一意性**：従来の`u0`など録音をまたぐ同一IDを廃止し、録音接続ごとにランダムな識別子を付与。録音2回目の字幕が履歴の重複除去で消える不具合を防止
- **翻訳の正しい紐付け**：Bedrock翻訳が遅れて届いても、再録音した別の会議や別の発話に誤って上書きしないよう、同一のセッション付き発話IDで履歴と照合
- **CPU競合の抑制**：複数のブラウザーセッションが同時接続しても、`ASR_MAX_CONCURRENT_INFERENCES=1`（デフォルト）により同時Whisper推論を1件に制限。4 vCPU環境で過負荷を抑える狙い
- **本番負荷時の優先制御**：音声入力キューが`ASR_INTERIM_BACKLOG_THRESHOLD=8`件以上に増えた場合、暫定字幕の再認識だけ省略し、確定字幕と音声保持を優先
- **評価と実装の同期**：`replay_asr.py`も同じ100msの発話境界判定と音声イベント時刻を使用

`ASR_MAX_CONCURRENT_INFERENCES` はCPU数と同時接続数を考慮して、変更時に負荷を実測してください。大きい数ほど高精度になる設定ではありません。高負荷時はサーバーの上限32パケットに到達すると既存の過負荷保護が働き、ブラウザー接続は終了します。**実環境での精度・遅延・過負荷挙動は引き続き未検証**です。

## セキュリティ方針（PoC）

- 従来のGoogle翻訳とOpenRouter要約は**削除**。翻訳・AI要約の文字起こしテキスト送信先はAmazon Bedrock（Converse API）のみ。モデル未設定では両機能を無効化
- WebSocket接続時、ブラウザーのOriginとHostを照合し、最初のメッセージで`APP_ACCESS_TOKEN`を検証（未設定なら拒否）
- アクセストークンは環境変数/Secrets Managerから取得。GitやDockerイメージに埋め込まない
- バイナリ1チャンク最大128 KiB、メモリキュー32件（通常0.5秒/件、最大約16秒のPCM）。過負荷時は接続を終了
- Whisperに渡す音声はプロセスメモリ内で処理。S3への保存機能は未実装
- 要約API `/api/summarize` はPoCトークンによるHTTP Bearer認証。要約の入力件数/文字数にも上限を設定
- 翻訳はユーザーがチェックをONにした時のみ、要約はボタンを押した時のみBedrockへ文字起こしテキストを送信（音声は送信しない）
- `/healthz` はNLB/ECSのヘルスチェック用。認証不要だがネットワーク上のアクセスを制限する

**このトークンはPoC用の共有シークレットであり、会社のSSO/個人認証の代わりにはなりません。** 本番利用時は企業の認証基盤（例: Entra ID、認証済みリバースプロキシ等）を統合し、監査・会議同意・保持ポリシーを確認してください。社内限定NLBであっても暗号化（WSS/TLS）とECS Security Groupを確認してください。現行実装にはユーザー別アクセス制御・接続数制限がありません。

## 静的チェック

依存ライブラリをインストールせずにPython構文・ECS JSON・安全設定の最低限を検証できます。Node.jsがあればブラウザーJavaScriptの構文もチェックします。

```bash
python deploy/fargate/verify.py
# numpyがインストール済みなら、合成音声のみのユニットテストを実施
cd deploy/fargate
python -m unittest -v test_streaming_asr
# requirements.txtの依存がインストール済みならBedrockのモックテストも実行可
python -m unittest -v test_bedrock_text
# Node.jsがあれば16k/48kリサンプリングのJSテストも実施
node test_audio_worklet.js
# 連続録音の字幕ID・遅延翻訳の紐付け・暫定表示をオフライン検証
node test_subtitle_ui.js
# 正解テキストとの比較評価（モデル不要）
python -m unittest -v test_accuracy_metrics test_vexa_quality test_replay_asr
# モデルを使ってCER / 専門用語再現率 / リアルタイム倍率を測る場合
python benchmark_asr.py --manifest C:/evaluation/manifest.jsonl --model-path C:/models/small --terms-file C:/evaluation/terms.txt
# 発話境界・字幕暫定確定も含めたCPU replay評価（オフライン）
python replay_asr.py --manifest C:/evaluation/manifest.jsonl --model-path C:/models/small
```

## 精度評価：会議データを外部送信しない方法

`benchmark_asr.py` は**ローカルの16kHz・mono・16bit PCM WAV**と正解の文字起こしを使ってWhisperを測定します。ファイルも文字起こしも外部へ送信せず、モデルはローカルのディレクトリからのみ読み込みます。

評価データは利用許可済みで匿名化した音声を使い、Gitにコミットしないでください。JSONL 1行は次の形式です。

```json
{"id":"sample-001","wav":"audio/sample-001.wav","reference":"生産計画を確認します。","language":"ja"}
```

出力はCER（文字誤り率、低いほど良い）、専門用語再現率（用語が出現する正解文に対する包含率）、RTF（推論秒数 / 音声秒数、1未満ならリアルタイムより速い）の集計です。**単語の誤挿入や数値の誤りも別途検証してください。** このスクリプトが自動的に精度を向上させるわけではありません。

日本語CERはNFKC・小文字化・空白・句読点除去後に算出します。繰り返し推論のバラつきがあるため、同一音源・同一環境で従来設定と新設定を比較してください。実データでの精度値はまだ得られていません。

## Amazon Bedrockで翻訳・要約する（オプション）

Whisperによる**文字起こしは完全にコンテナ内**、Bedrockには**文字起こし済みのテキストのみ**を送信します。従来のGoogle翻訳・OpenRouterはFargate版では利用しません。

- `BEDROCK_MODEL_ID`: 空欄なら**翻訳・要約とも無効**。AWS環境で利用許可済みのモデルIDまたは推論プロファイルIDを設定（例は固定しません。利用可能なIDはアカウント・リージョン・契約で異なります）
- `BEDROCK_REGION`: Bedrock Runtimeを呼び出すリージョン。デフォルトは`ap-northeast-1`
- 必要IAM権限: **ECS task role** に対象のモデル/推論プロファイルに対する `bedrock:InvokeModel`。実行ロール（image pull用）ではなくタスクロールへ設定
- Private subnetからBedrockへ外向き通信する場合、NAT経由か `com.amazonaws.ap-northeast-1.bedrock-runtime` のインターフェースVPCエンドポイント（Private DNS有効）を設置。Endpoint用SGはFargate SGからの443を許可
- クロスリージョン推論プロファイルを使う場合、送信テキストが別のAWSリージョンへルーティングされる可能性があるため、社内データポリシーで許可リージョンを確認。必要な権限も利用プロファイルのルーティング先に合わせて設定
- Bedrockに接続してもGoogle翻訳/OpenRouterには送信されません。ただしCloudWatch等のログ・ネットワーク・モデルの利用許可は別途確認してください。AWS側の推論利用料金が発生します
- モデルIDを設定しても、初期状態では翻訳チェックがOFF。要約もボタン操作をした時のみ送信します

AWS公式：[Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html) / [Bedrock VPC endpoints](https://docs.aws.amazon.com/bedrock/latest/userguide/vpc-interface-endpoints.html)。

### ローカル実行でBedrockの動作を確認する

ローカル端末にAWS認証情報を用意済みの場合のみ、コンテナへAWSの認証情報を適切に渡して `BEDROCK_MODEL_ID` を設定してください。**アクセスキーをDockerfileやGitに埋め込まない**こと。Fargateではタスクロールを使うためアクセスキーは不要です。

## ローカル起動

Dockerとインターネット接続（**ビルド時のみ**モデル取得）が必要です。

```bash
# リポジトリルートから
docker build --platform linux/amd64 -t translate-whisper:small ./deploy/fargate
docker run --rm --platform linux/amd64 -p 127.0.0.1:8000:8000 \
  -e APP_ACCESS_TOKEN='REPLACE_WITH_LONG_RANDOM_SECRET' \
  --name translate-whisper translate-whisper:small
```

ブラウザーで `http://localhost:8000/` を開き、上記トークンを入力してください。モデルが起動時にロードされたあと `http://localhost:8000/healthz` が `{"status":"ok"}` を返します。実環境はHTTPS/WSSで利用します。

### モデルサイズ

デフォルトは `small`（多言語）。軽量化する場合は `--build-arg MODEL_SIZE=base` を指定できます。x86_64 FargateではCPU/int8を利用し、GPUは使えません。メモリ・レイテンシ・リアルタイム性は実機で測定してください。

## ECRへ登録（例）

以下は自分のAWSアカウント・リージョン・リポジトリ名に読み替えます。既存のECRを使う場合、`create-repository` は不要です。

```bash
export AWS_REGION=ap-northeast-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/translate-whisper"

aws ecr describe-repositories --repository-names translate-whisper --region "$AWS_REGION" \
  || aws ecr create-repository --repository-name translate-whisper --region "$AWS_REGION"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker tag translate-whisper:small "$ECR_URI:small-poc"
docker push "$ECR_URI:small-poc"
```

本番運用では再現性のためイメージタグだけでなくECRのdigest固定も検討してください。例の`Dockerfile`は依存ライブラリの互換範囲を指定していますが、完全固定（ロックファイル・イメージダイジェスト）は次の段階です。

## ECS / NLBに接続

1. `ecs-task-definition.example.json` のプレースホルダーを**実際のARN/ECR URI**に置換し、ロググループ `/ecs/translate-whisper` とIAM実行ロールを事前作成する。
2. Secrets Managerに`APP_ACCESS_TOKEN`用の**プレーンテキストのシークレット値**を登録する。ECS実行ロールにその値の参照許可を設定する。
3. タスク定義を登録し、既存のECSクラスターに**既存フロントエンドとは別のECSサービス**として配置する。まずは4 vCPU / 8 GiB / 1タスクで測定する。
4. Fargateはprivate subnet/awsvpc/タスクSGでNLBターゲットからのTCP 8000だけを許可する。Fargateのpublic IPは付与しない。
5. 既存のNLBに**別ターゲットグループ（ターゲット種別 `ip`、ポート8000）**を作成し、対象サービスへ紐付ける。NLB単体ではHTTPパス単位の振り分けができないため、既存アプリとの共存は**別のリスナー/ポート/別NLB**または既存HTTPリバースプロキシを使う。既存ターゲットグループの置換はしない。
6. NLBの443 TLSリスナーを利用する場合、NLB→Fargate間はターゲットグループ設定次第で平文TCPとなる。会議音声の扱いに応じて内部TLSを設計・確認する。ブラウザー側は必ずHTTPS/WSS。
7. NLBターゲットグループはHTTP `/healthz` をヘルスチェックに設定し、モデルロード時間に十分な猶予を取る。長時間WebSocketを維持するならNLBのアイドルタイムアウトや再接続設計を実環境で検証する。
8. ECR Pull/CloudWatch Logs/Secrets Manager利用のため、Private subnetからNATまたは必要なVPC Endpointへの疎通を確認する。

既存NLBに別サービスを追加できるかは、リスナー構成と既存アプリのプロキシ有無次第です。**このPRにはAWSへの自動デプロイ処理は含めません。**

## PoC確認項目

- [ ] `docker build` が成功し `/healthz` が200を返す
- [ ] BedrockのConverse API翻訳・要約を承認済みモデルで実機検証
- [ ] ブラウザーが正しいトークンで接続し、誤ったトークンでは音声を送れない
- [ ] WSSで0.5秒チャンクが送信され、認識結果が返る
- [ ] Teamsの相手音声が取れるか（特にデスクトップ版）をWindows/Edgeなど実機で確認
- [ ] 文字起こし遅延、CPU/メモリ使用率、キュー満杯時の動作を確認
- [ ] Vexa参考の暫定下線表示、発話タイムスタンプ、3秒無音後の文脈リセット、低確度判定を同じ音源で検証
- [ ] replay_asr.py と benchmark_asr.py でCER/RTFを比較し、CER悪化やCPU過負荷なら変更を調整・ロールバック
- [ ] 0.5秒以下の発話・語頭の小声・無音・連続発話・Stop直後の字幕確定を確認
- [ ] 変更前後の同一会議音声でCER、専門用語・数字正答率、発話欠落、P50/P95字幕遅延を測定
- [ ] 翻訳ONでもWhisper処理がBedrock応答待ちに塞がれないことを確認
- [ ] モデル未設定時に翻訳・要約が無効であることを確認
- [ ] モデル設定時にGoogle翻訳・OpenRouterには一切接続せず、Bedrock Runtimeだけが必要になることを確認
- [ ] IAMロールとPrivateLink、リージョン、Bedrockのモデルアクセス権を実環境で検証
- [ ] 自分のマイク＋相手音声を同時取得したい場合は別途ミキシング実装を追加

## サイズ・コスト

Whisper smallのモデルファイル単体は数百MBですが、ECRの実イメージ容量は依存パッケージと圧縮方法によって変わります。**サイズはdocker build後に`docker image inspect`で実測してください。** Fargate Linux 1.4.0+ はデフォルト20 GiBのエフェメラルストレージを備えます。圧縮・展開後のコンテナイメージもここに計上されます。

## License and third-party notices

The original application source is under the **MIT License**. This Fargate
image carries its own [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for clarity when the image is
redistributed.

MIT permits commercial use, modification and redistribution with the original
copyright and license notice retained. **The model, Python/native packages and
base image are separately licensed.** Review their exact installed revisions
and redistribution obligations; the notice file is an upstream summary, not a
complete license inventory.
