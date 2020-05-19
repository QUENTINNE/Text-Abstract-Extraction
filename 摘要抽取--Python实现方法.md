# 摘要抽取--Python实现

## 1. HanLP

> pyhanlp: Python interfaces for HanLP1.x
>
> [官方地址](https://github.com/hankcs/pyhanlp)

input:

```python
from pyhanlp import *

# 关键词提取
document = "水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
           "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
           "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
           "严格地进行水资源论证和取水许可的批准。"
# 自动摘要
print(HanLP.extractSummary(document, 3))
```

output:

```python
[严格地进行水资源论证和取水许可的批准, 水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露, 有部分省超过红线的指标]
```

**结果特点：**

1. 参数只有一个。抽取摘要的句子数量。
2. 结果乱序。和原文本顺序不一致。
3. [（文本遇到标点符号会自动断句，使得结果不够好）](https://github.com/hankcs/HanLP/issues/990)

**使用缺点：**

1. 依赖包一个，还有Java环境等。不适用于离线环境安装。

## 2. snownlp

> SnowNLP: Simplified Chinese Text Processing
>
> [官方网站](https://github.com/isnowfy/snownlp)

input:

```python
from snownlp import SnowNLP

text = u"水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
       "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
       "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
       "严格地进行水资源论证和取水许可的批准。"
s = SnowNLP(text)
print(s.summary(3))
```

output:

```python
['水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露', '有部分省超过红线的指标', '陈明忠表示']
```

结果特点：

1. 参数唯一。
2. 结果乱序。
3. 遇到符号""不会乱序，识别为整句。
4. 单独的包，可以离线安装。

## 3. jiagu



```python
import jiagu

text = u"水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
       "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
       "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
       "严格地进行水资源论证和取水许可的批准。"
summarize = jiagu.summarize(text, 3)  # 摘要
print(summarize)
```

```python
['对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，严格地进行水资源论证和取水许可的批准。', '水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，有部分省超过红线的指标。', '']
```

**结果特点：**

1. 参数唯一。
2. 结果乱序。
3. 遇到符号""不会乱序。
4. 单独的包，可以离线安装。
5. 带句号才认为是一个句子。

## 4. TextRank4ZH

[官方网站](https://github.com/letiantian/TextRank4ZH)

[作者说明TextRank的文章](https://www.letianbiji.com/machine-learning/text-rank.html)

```python
from textrank4zh import TextRank4Sentence

text = u"水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
       "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
       "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
       "严格地进行水资源论证和取水许可的批准。"
tr4s = TextRank4Sentence()
tr4s.analyze(text=text, lower=True, source='all_filters')

for item in tr4s.get_key_sentences(num=3):
    print(item.sentence)
```

```python
水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，有部分省超过红线的指标
对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，严格地进行水资源论证和取水许可的批准
```

结果特点：

1. 多个参数。

   > text --  文本内容，字符串。
   > lower--  是否将文本转换为小写。默认为False。
   > source       --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来生成句子之间的相似度。默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。
   > sim_func     --  指定计算句子相似度的函数。
   >
   > -----------------------------------------------------------
   >
   > sentences：由句子组成的列表。
   > words_no_filter：对sentences中每个句子分词而得到的两级列表。
   > words_no_stop_words：去掉words_no_filter中的停止词而得到的二维列表。
   > words_all_filters：保留words_no_stop_words中指定词性的单词而得到的二维列表。
   >
   > 
   >
   > sentence_min_len = 6
   >
   > """获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要。

2. 结果乱序。以句号为整句。

## 5. FastTextRank

[官方网站](https://github.com/ArtistScript/FastTextRank)



```python
from FastTextRank.FastTextRank4Sentence import FastTextRank4Sentence

mod = FastTextRank4Sentence(use_w2v=False, tol=0.0001)

text = u"水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
       "根据刚刚完成了水资源管理制度的考核。有部分省接近了红线的指标，" \
       "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
       "严格地进行水资源论证和取水许可的批准。"
summarize = mod.summarize(text, 2)  # 摘要
print(summarize)
```

```python
['对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，严格地进行水资源论证和取水许可的批准。', '有部分省接近了红线的指标，有部分省超过红线的指标。']
```

结果特点：

1. 多个参数可选择。

   > :param use_stopword: 是否使用停用词
   > :param stop_words_file: 停用词文件路径
   > :param use_w2v: 是否使用词向量计算句子相似性
   > :param dict_path: 词向量字典文件路径
   > :param max_iter: 最大迭代伦茨
   > :param tol: 最大容忍误差

2. 结果乱序。以句号作为整句。

## 6. macropodus

[官方网站](https://github.com/yongzhuo/Macropodus)

```python
import macropodus

text = u"水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
       "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
       "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
       "严格地进行水资源论证和取水许可的批准。"
sum = macropodus.summarize(text)
for i in range(len(sum)):
    print(sum[i][1])
```

```python
对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，严格地进行水资源论证和取水许可的批准
水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，有部分省超过红线的指标
```

1. 参数可调。

   > 文本摘要(summarization, 可定义方法, 提供9种文本摘要方法, 'lda', 'mmr', 'textrank', 'text_teaser')
   >
   > sents = macropodus.summarization(text=summary, type_summarize="lda")

2. 以句号为整句。