# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 19:31
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : data_helper.py
# @Software: PyCharm


'''
option:创建字典
function：jiebea

'''

import re
import os
import jieba
import pickle
import numpy as np
# 获取停用词列表
def getStopWords(txt_path='./data/stopWords.txt'):
	stopWords = []
	with open(txt_path, 'r') as f:
		for line in f.readlines():
			stopWords.append(line[:-1])
	return stopWords


# 把某list统计进dict
def list2Dict(wordsList, wordsDict):
	for word in wordsList:
		if word in wordsDict.keys():
			wordsDict[word] += 1
		else:
			wordsDict[word] = 1
	return wordsDict


# 获取文件夹下所有文件名
def getFilesList(filepath):
	return os.listdir(filepath)


# 统计某文件夹下所有邮件的词频
def wordsCount(filepath, stopWords, wordsDict=None):
	if wordsDict is None:
		wordsDict = {}
	wordsList = []
	filenames = getFilesList(filepath)
	for filename in filenames:
		with open(os.path.join(filepath, filename), 'r') as f:
			for line in f.readlines():
				# 过滤非中文字符
				pattern = re.compile('[^\u4e00-\u9fa5]')
				line = pattern.sub("", line)
				words_jieba = list(jieba.cut(line))
				for word in words_jieba:
					if word not in stopWords and word.strip != '' and word != None:
						wordsList.append(word)
		wordsDict = list2Dict(wordsList, wordsDict)
	return wordsDict


# 保存字典类型数据
def saveDict(dict_data, savepath='./results.pkl'):
	with open(savepath, 'wb') as f:
		pickle.dump(dict_data, f)


# 读取字典类型数据
def readDict(filepath):
	with open(filepath, 'rb') as f:
		dict_data = pickle.load(f)
	return dict_data

if __name__ == '__main__':
    #  Part1
    stopWords = getStopWords(txt_path='./data/stopWords.txt')
    wordsDict = wordsCount(filepath='./data/normal', stopWords=stopWords)
    wordsDict = wordsCount(filepath='./data/spam', stopWords=stopWords, wordsDict=wordsDict)
    saveDict(dict_data=wordsDict, savepath='./results.pkl')