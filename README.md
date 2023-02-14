# pytorch_bert_chinese_ner
基于bert的中文实体识别，并使用字形嵌入和拼音嵌入。

# 依赖

```python
transformers==4.5.0
torch==1.6.0+
pypinyin
```

# 运行

1、hugging face上下载模chinese-bert-wwm-ext模型文件到model_hub/chinese-bert-wwm-ext/下。

2、需要生成字形嵌入所需的文件，运行：```python get_glyph.py```

3、训练、验证、测试和预测。

单纯的bert实体识别：

```python
python main.py

【eval】 precision=0.9651 recall=0.9511 f1_score=0.9580
['ORG', 'TITLE', 'CONT', 'NAME', 'PRO', 'LOC', 'RACE', 'EDU']
          precision    recall  f1-score   support

     ORG       0.95      0.93      0.94       551
   TITLE       0.95      0.95      0.95       762
    CONT       1.00      1.00      1.00        28
    NAME       1.00      1.00      1.00       112
     PRO       1.00      0.94      0.97        33
     LOC       1.00      1.00      1.00         6
    RACE       1.00      1.00      1.00        14
     EDU       0.98      0.98      0.98       112

micro-f1       0.96      0.95      0.95      1618

text= 陈学军先生：1967年5月出生，大学毕业，高级经济师。
{'ORG': [], 'TITLE': [('高级经济师', 21)], 'CONT': [], 'NAME': [('陈学军', 0)], 'PRO': [], 'LOC': [], 'RACE': [], 'EDU': [('大学', 16)]}
```

加上字形嵌入和拼音嵌入：

```python
python main2.py

【eval】 precision=0.9591 recall=0.9501 f1_score=0.9546
【best f1】 0.9546
['ORG', 'TITLE', 'CONT', 'NAME', 'PRO', 'LOC', 'RACE', 'EDU']
          precision    recall  f1-score   support

     ORG       0.93      0.93      0.93       551
   TITLE       0.95      0.94      0.95       762
    CONT       1.00      1.00      1.00        28
    NAME       0.99      1.00      1.00       112
     PRO       0.97      0.86      0.91        33
     LOC       0.67      0.80      0.73         6
    RACE       0.93      0.93      0.93        14
     EDU       0.97      0.94      0.96       112

micro-f1       0.95      0.94      0.94      1618

text= 陈学军先生：1967年5月出生，大学毕业，高级经济师。
{'ORG': [], 'TITLE': [('高级经济师', 21)], 'CONT': [], 'NAME': [('陈学军', 0)], 'PRO': [], 'LOC': [], 'RACE': [], 'EDU': [('大学', 16)]}
```

# 总结

1、加入字形和拼音，训练速度没有原始的快，而且效果也没原始的好，可能的原因是：

- 不适合该数据集；
- 可能需要调整参数或者训练策略；

2、如果想要训练其他的数据集，可参考resume数据集的格式。

# 参考

部分代码参考ChineseBert。
