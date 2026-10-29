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
# 正解テキストとの比較評価（モデル不要）
python -m unittest -v test_accuracy_metrics
# モデルを使ってCER / 専門用語再現率 / リアルタイム倍率を測る場合
python benchmark_asr.py --manifest C:/evaluation/manifest.jsonl --model-path C:/models/small --terms-file C:/evaluation/terms.txt
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
- [ ] 0.5秒以下の発話・語頭の小声・無音・連続発話・Stop直後の字幕確定を確認
- [ ] 変更前後の同一会議音声でCER、専門用語・数字正答率、発話欠落、P50/P95字幕遅延を測定
- [ ] 翻訳ONでもWhisper処理がBedrock応答待ちに塞がれないことを確認
- [ ] モデル未設定時に翻訳・要約が無効であることを確認
- [ ] モデル設定時にGoogle翻訳・OpenRouterには一切接続せず、Bedrock Runtimeだけが必要になることを確認
- [ ] IAMロールとPrivateLink、リージョン、Bedrockのモデルアクセス権を実環境で検証
- [ ] 自分のマイク＋相手音声を同時取得したい場合は別途ミキシング実装を追加

## サイズ・コスト

Whisper smallのモデルファイル単体は数百MBですが、ECRの実イメージ容量は依存パッケージと圧縮方法によって変わります。**サイズはdocker build後に`docker image inspect`で実測してください。** Fargate Linux 1.4.0+ はデフォルト20 GiBのエフェメラルストレージを備えます。圧縮・展開後のコンテナイメージもここに計上されます。
