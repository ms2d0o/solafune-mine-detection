# Introduction
- solafuneの[採掘場検知コンペ](https://solafune.com/ja/competitions/58406cd6-c3bb-4f7a-85c7-c5a1ad67ca03?menu=about&tab=overview)用のリポジトリです

## Python setting
poetry,pyenvはインストール済みの環境を想定しています。
```
poetry install
```

## training
```
poetry run python scripts/run.py
```

DEBUGをオフにする
```
poetry run python scripts/run.py DEBUG=False
```
