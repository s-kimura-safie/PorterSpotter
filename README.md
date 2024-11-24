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
以下のコマンドでDockerイメージを作成します。
```bash
./docker/x86-64/build.sh
```

次のコマンドでdocker環境に入ります。
```bash
./docker/x86-64/run.sh
```

Docker環境で以下を実行します。
```bash
make
```

### ビルド (EVK)
以下のコマンドでDockerイメージを作成します。
```bash
./docker/aarch64/build.sh
```

次のコマンドでdocker環境に入ります。
```bash
./docker/aarch64/run.sh
```

Docker環境で以下を実行します。
```bash
make TARGET=aarch64
```

### 推論モデル
以下の2つのファイルを`./models`に格納してください。  
- 姿勢推定モデル：[こちら](https://drive.google.com/file/d/13cT1FtoMZ7mRD3-Me9qZJP0hl5TyAz1P/view?usp=drive_link)のRTMPoseのdlcファイル

## 実行（x86-64）

### OfflinePoseEstimator
オフラインで画像からポーズ推定を行います。

`./images`に入力する画像を保存してください。  
以下のコマンドを実行すると検出した人物に対して姿勢推定された画像が `outputs` に出力されます。
```bash
./bin/x86-64/OfflinePoseEstimator -d models/yolov5s.dlc -p models/rtmpose.dlc images/*.jpg
```

### OfflineVideoAnalysis
オフラインで動画から、200ms間隔で解析し、人の動作（転倒を含む）を認識します。

`./videos`に入力する動画を保存してください。
以下のコマンドを実行すると検出した人物に対してポーズのkeypointおよび検出バウンディングボックスを描画した動画が `output` に出力されます。
```bash
./bin/x86-64/OfflineVideoAnalysis --d models/yolov5s.dlc --h models/rtmpose.dlc --a ./models/stgcn.dlc --input_video videos/sample.mp4 --output_dir output --output_video --person_box --skeleton
```

## 実行（EVK）

### OfflinePoseEstimator
実行する手順がEdgeObjectRecogと同様です。[参照](https://github.com/SafieDev/EdgeObjectRecog/tree/main/standalone#%E5%AE%9F%E8%A1%8Cevk)

### OfflineEvkVideoAnalysis
`Makefile`の`MAIN_SRCS`を下記のように修正してください。
```cpp
- MAIN_SRCS		:= OfflinePoseEstimator.cpp OfflineVideoAnalysis.cpp 
+ MAIN_SRCS		:= OfflinePoseEstimator.cpp OfflineEvkVideoAnalysis.cpp 
```
実行手順は`OfflinePoseEstimator`と同じです。

