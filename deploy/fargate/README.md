# ECS Fargate版（CPU Whisper）

本ディレクトリはリポジトリ直下の `app.py` / `templates/index.html` をコピーして、社内ネットワーク向けPoCとして独立したものです。**元のアプリは変更していません。**

## できること

- ブラウザーのマイク、または共有画面のシステム音声を取得し、16 kHz / 16-bit PCM（2秒ごと）をWebSocketで送信
- NLB → FargateのFastAPI `/ws/audio` が受信し、**同じPythonプロセスのfaster-whisper（CPU）** で認識
- 同じWebSocket経由で文字起こしを返却。Whisper用の別APIやGPUは不要
- モデルはイメージのビルド時に取得して同梱。実行時はモデルの外部ダウンロード不要

**重要:** Teams標準の文字起こしを取得するものではありません。PCで共有できるTeams音声をブラウザー経由で再認識します。システム音の取得可否はOS・Teams（デスクトップ/ブラウザー）・ブラウザーの仕様に依存します。また今のUIではマイク/システム音は二者択一であり、会議全員の発言取得は保証しません。

## セキュリティ方針（PoC）

- 従来のGoogle翻訳とOpenRouter要約は**コード側で無効化**し、該当ライブラリを依存から除外
- WebSocket接続時、ブラウザーのOriginとHostを照合し、最初のメッセージで`APP_ACCESS_TOKEN`を検証（未設定なら拒否）
- アクセストークンは環境変数/Secrets Managerから取得。GitやDockerイメージに埋め込まない
- バイナリ1チャンク最大128 KiB、メモリキュー8件。過負荷時は接続を終了
- Whisperに渡す音声はプロセスメモリ内で処理。S3への保存機能は未実装
- `/healthz` はNLB/ECSのヘルスチェック用。認証不要だがネットワーク上のアクセスを制限する

**このトークンはPoC用の共有シークレットであり、会社のSSO/個人認証の代わりにはなりません。** 本番利用時は企業の認証基盤（例: Entra ID、認証済みリバースプロキシ等）を統合し、監査・会議同意・保持ポリシーを確認してください。社内限定NLBであっても暗号化（WSS/TLS）とECS Security Groupを確認してください。現行実装にはユーザー別アクセス制御・接続数制限がありません。

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
- [ ] ブラウザーが正しいトークンで接続し、誤ったトークンでは音声を送れない
- [ ] WSSで2秒チャンクが送信され、認識結果が返る
- [ ] Teamsの相手音声が取れるか（特にデスクトップ版）をWindows/Edgeなど実機で確認
- [ ] 文字起こし遅延、CPU/メモリ使用率、キュー満杯時の動作を確認
- [ ] Dockerビルド時を除き、外部翻訳・要約へ会議データを送らないことを通信ログで確認
- [ ] 自分のマイク＋相手音声を同時取得したい場合は別途ミキシング実装を追加

## サイズ・コスト

Whisper smallのモデルファイル単体は数百MBですが、ECRの実イメージ容量は依存パッケージと圧縮方法によって変わります。**サイズはdocker build後に`docker image inspect`で実測してください。** Fargate Linux 1.4.0+ はデフォルト20 GiBのエフェメラルストレージを備えます。圧縮・展開後のコンテナイメージもここに計上されます。
