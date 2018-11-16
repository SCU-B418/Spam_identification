# Spam-identification
目标：识别中文的垃圾邮件

# 参考解决方法

|文件名|作者|效果|说明|
|-|-|-|-|
|[Spam_identification_solution(Keras).py](Spam_identification_solution(Keras).py)|袁宵|98%-100%|用神经网络框架 Keras 设计了两种简单的神经网络：卷积神经网络和循环神经网络，较好的解决了此垃圾邮件二分类问题|
|[Spam_identification_solution(Keras).ipynb](Spam_identification_solution(Keras).ipynb)|袁宵|98%-100%|与[Spam_identification_solution(Keras).py](Spam_identification_solution(Keras).py) 内容一致|

相关模块： 
----------------  
scikit-learn模块；  
jieba模块；  
numpy模块；  
以及一些Python自带的模块。  
  
环境搭建：  
----------------
python3   
   
数据描述：  
----------------
【训练数据集】：  
7063封正常邮件(data/normal文件夹下)   
7775封垃圾邮件(data/spam文件夹下)   
【测试数据集】：  
共392封邮件(data/test文件夹下)。  
  
c一种实现步骤：  
----------------
1）【创建字典】 1人     
方法：正则表达式，jieba分词库  
提示：需要清除一些停用词  
  
2）【特征提取】 1人    
方法：把每封信的内容转换为词向量，每一维代表一个高频词在该封信中出现的频率，最后得到一个特征向量矩阵  
  
3）【训练分类器】 2人    
1.线性分类  
2.逻辑斯提回归  
3.svm  
4.贝叶斯  
提示：模型需要序列化存储  
  
4）【性能测试】 1人  
利用模型预测结果，并绘图，得出混淆矩阵   
提示：matplotlib  
