# Porter Spotter

## 概要
人物が特定のものを運んでいることを検知するアプリケーションです。

## 使用した手法
| Module | algorithm | Reference Implementation
| --- | ----------- |-----
| Detection | Yolov5-s | EdgeObjectRecog
| Tracking | Byte | EdgeObjectRecog
| Pose estimation | RTMPose-t | [mmpose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)

## 環境構築

### SDK
以下のファイルを[google drive](https://drive.google.com/drive/folders/1-EoS-5tb6C_o_Mv5DT107SuXkUN8cH_W) から取得し、`./sdk`においてください。
- snpe-1.68.0.zip
- android-ndk-r17c-linux-x86_64.zip (EVK使用時)

### ライブラリ
EVKを使う場合、[OpenCVライブラリ](https://drive.google.com/drive/folders/14S_pn7pJF18ZJoPApeReiHndeNvIlDr2) を `./lib/aarch64-oe-linux-gcc8.2/libopencv_world.so` にコピーしてください。

### ビルド (x86)
以下のコマンドでDocker環境でビルドします。
```bash
./docker/x86-64/build.sh
./docker/x86-64/run.sh
make
```

### 推論モデル
以下の2つのファイルを`./models`に格納してください。  
- 物体検出モデル：【追加予定】
- 姿勢推定モデル：[こちら](https://drive.google.com/file/d/13cT1FtoMZ7mRD3-Me9qZJP0hl5TyAz1P/view?usp=drive_link)のRTMPoseのdlcファイル

## 実行（x86-64）

### OfflinePoseEstimator
オフラインで画像からポーズ推定を行います。

`./images`に入力する画像を保存してください。  
以下のコマンドを実行すると検出した人物に対して姿勢推定された画像が `outputs` に出力されます。
```bash
./bin/x86-64/OfflinePoseEstimator -d models/yolov5s.dlc -p models/rtmpose.dlc images/*.jpg
```

### OfflinePorterSpotter
オフラインで画像から人物が対象の物体を持っているかを検出します。
`./images`に入力する画像を保存してください。  
以下のコマンドを実行すると検出した人物が対象の物体を持っているかのを判定した結果が `outputs` に出力されます。
```bash
./bin/x86-64/OfflinePorterSpotter -d models/yolov8s.dlc -p models/rtmpose.dlc -input_dir images -person_box -object_box -skeleton
```

### OfflineVideoPorterSpotter
オフラインで動画上の人物が対象の物体を持っているかを検出します。
`./videos`に入力する動画を保存してください。  
以下のコマンドを実行すると検出した人物が対象の物体を持っているかのを判定した結果が `outputs` に出力されます。
```bash
./bin/x86-64/OfflineVideoPorterSpotter -d models/yolov8s.dlc -p models/rtmpose.dlc -input_file videos/sample.mp4 -output_video -person_box -object_box -skeleton
```
