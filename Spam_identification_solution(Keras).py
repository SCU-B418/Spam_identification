
# coding: utf-8

# # Spam_identification_solution

# ## 作者：袁宵
# ## 时间：2018/11/16

# # 导入依赖库

# In[1]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPool1D
from tensorflow.keras.utils import plot_model
import numpy as np

import os
import chardet
import re
import jieba


# ## 超参数

# In[2]:


epochs = 3
batch_size = 64
max_sequence_length = 150
embedding_size = 16
num_words = 2000
learning_rate = 0.001

# RNN 参数
LSTM_unit = 32

# CNN 参数
filters_numbers = 32
kernel_size = 5


# # 数据预处理

# ### 查看文件编码

# In[3]:


a_file_path = 'data/spam/2829'

def find_text_encode(a_file_path):
    with open(a_file_path, 'rb') as f:
        return chardet.detect(f.read())

file_info = find_text_encode(a_file_path)

file_encoding = file_info['encoding']
print(file_encoding)


# ### 清洗文件内容，获取以空格分隔的中文字符串数据

# In[4]:


# 过滤非中文字符
pattern = re.compile('[^\u4e00-\u9fa5]')


# In[5]:


spam_filepath = os.path.join("data", "spam")
normal_filepath = os.path.join("data", "normal")
print(spam_filepath)


# In[6]:


def get_clean_sentence(filepath, file_encoding):
    fail_file_names_list = []
    data = []
    filenames = os.listdir(filepath)
    for filename in filenames:
        file_chinese_content = []
        try:
            with open(os.path.join(filepath, filename), encoding=file_encoding) as f:
                for line in f.readlines():
                    line = pattern.sub("", line)
                    line_cut = jieba.cut(line)
                    file_chinese_content.extend(list(line_cut))
        except:
            fail_file_names_list.append(filename)  
        if len(file_chinese_content) > 0:
            file_chinese_content_sequence = " ".join(file_chinese_content)
            data.append(file_chinese_content_sequence)
    return data, fail_file_names_list


# In[7]:


spam_data, spam_fail_file_names_list = get_clean_sentence(spam_filepath, file_encoding)
normal_data, normal_fail_file_names_list = get_clean_sentence(normal_filepath, file_encoding)


# In[8]:


print(len(spam_data), len(normal_data))


# ### 创建字典

# In[9]:


tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(spam_data + normal_data)

len(tokenizer.index_word)


# ### 序列化数值化 把中文转换成整数值

# In[10]:


def serialization_numeralization(spam_data, normal_data):
    x_train_spam = tokenizer.texts_to_sequences(spam_data)
    x_train_normal = tokenizer.texts_to_sequences(normal_data)

    x_train_spam_pad = pad_sequences(x_train_spam, max_sequence_length)
    x_train_normal_pad = pad_sequences(x_train_normal, max_sequence_length)

    x_train = []
    y_train = []
    for it in x_train_normal_pad:
        x_train.append(it)
        y_train.append(0)
    for it in x_train_spam_pad:
        x_train.append(it)
        y_train.append(1)
    return x_train, y_train


# In[11]:


x_train, y_train = serialization_numeralization(spam_data, normal_data)
print(len(x_train), len(y_train))


# # 设计模型 （以下模型二选一）

# ## CNN 模型 训练速度快

# In[12]:


model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=embedding_size, input_length=max_sequence_length))
model.add(Conv1D(filters=filters_numbers, kernel_size=kernel_size))
model.add(GlobalMaxPool1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

plot_model(model, to_file="Spam_identification_CNN_model.png", show_shapes=True)          


# ## LSTM 模型 训练速度慢
# 对于具有2个类的单输入模型（二进制分类）：

model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=embedding_size, input_length=max_sequence_length))
model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

plot_model(model, to_file="Spam_identification_LSTM_model.png", show_shapes=True)
# # 准备数据

# In[13]:


x_train_np = np.array(x_train)
y_train_np = np.array(y_train)
y_train_np = y_train_np[:, np.newaxis]


# # 训练模型

# In[14]:


model.fit(x_train_np, y_train_np,
          batch_size=batch_size, epochs=epochs,
          shuffle=True, validation_split=0.2)


# # 测试模型

# In[15]:


test_spam_filepath = os.path.join("data", "test","spam")
test_normal_filepath = os.path.join("data", "test","normal")


# In[16]:


test_spam_data, test_spam_fail_file_names_list = get_clean_sentence(test_spam_filepath, file_encoding)
test_normal_data, test_normal_fail_file_names_list = get_clean_sentence(test_normal_filepath, file_encoding)


# In[17]:


x_test, y_test = serialization_numeralization(test_spam_data, test_normal_data)
print(len(x_test), len(y_test))


# In[18]:


x_test_np = np.array(x_test)
y_test_np = np.array(y_test)
y_test_np = y_test_np[:, np.newaxis]


# In[19]:


model.evaluate(x_test_np, y_test_np)

