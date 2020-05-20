# -*- coding: utf-8 -*-
# @Time    : 2020/5/20 15:04
# @Author  : QUENTINNE
# @File    : main.py
# @Software: PyCharm

import datetime
import re

import pandas as pd
import xlrd
from snownlp import SnowNLP

# 时间参数
month = 2
year = 2019

# 文件路径
data = xlrd.open_workbook(r'D:\文件存放路径\样例数据\%s_%s.xlsx' % (year, month))
table = data.sheet_by_index(0)
cols = table.col_values(14)

# 正则匹配的规则（取出固定前缀和后缀中间的文本）
rule = r'C_JUDGEMENT_RESULT":"(.*?)","C_CASE_CAUSE'


# 正则匹配
def subString(template):
    slotList = re.findall(rule, template)
    return slotList


# 主要函数
def output(n):
    resultList = [[0] for i in range(len(cols))]  # 初始化原文书详情
    abstractList = [[0] for j in range(len(cols))]  # 初始化摘要
    for k in range(len(cols)):
        content = cols[k]
        resultList[k] = subString(content)
        if len(resultList[k]) == 1:  # 非空
            if len(resultList[k][0]) >= 5:  # 有效字符
                # sText = ''.join(resultList[k])
                sText = resultList[k][0]
                s = SnowNLP(sText).summary(n)  # 摘要
                abstractList[k] = s
        dfResult = pd.DataFrame(resultList).rename(columns={0: 'Result'})
        dfResult.to_csv('D://文件存放路径/结果数据/C_JUDGEMENT_RESULT_%s0%s.csv' % (year, month), index=False)
        dfAbstract = pd.DataFrame(abstractList).rename(columns={0: 'Abstract_1', 1: 'Abstract_2', 2: 'Abstract_3'})
        dfAbstract.to_csv('D://文件存放路径/结果数据/C_JUDGEMENT_RESULT_Abstract_%s0%s.csv' % (year, month),
                          index=False)


def main():
    n = 3  # 抽取摘要的句子数量
    startTime = datetime.datetime.now()
    output(n)
    print('一共%s条数据，已顺利全部抽取摘要!' % len(cols))
    endTime = datetime.datetime.now()
    print('所花时间：%ss' % (endTime - startTime).seconds)


if __name__ == '__main__':
    main()
