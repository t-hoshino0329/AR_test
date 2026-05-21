# EML 計算機 (EML Calculator)

論文 *"All elementary functions from a single operator"* に基づき、**eml(x, y) = exp(x) - ln(y)** と定数 **1** だけで初等関数を構成する関数電卓を実装しています。入力式はパース後に EML 木へコンパイルされ、結果とツリーを GUI で確認できます。

## 機能

- EML だけで四則演算・指数/対数・三角/双曲線/逆三角関数を構成
- 複素数対応の評価
- EML 木の可視化（Tkinter GUI）

## 実行方法

```bash
python eml_calc.py
```

※ Python と Tkinter が必要です（標準インストールに含まれます）。
