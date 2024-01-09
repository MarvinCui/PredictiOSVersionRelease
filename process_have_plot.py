import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置支持中文的字体，这里使用 macOS 中的 "Heiti TC"，根据您的操作系统和可用字体调整
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# 加载数据
data = pd.read_excel('Resources/iOS release date.xlsx')
data['days_since_first_release'] = (data['Date'] - data['Date'].min()).dt.days

# 识别每个大版本的首个小版本（假定为 x.0）
data['is_major'] = data['version number'].apply(lambda x: x.is_integer())
major_versions = data[data['is_major']]['version number']

latest_three_majors = major_versions.sort_values(ascending=False).head(3)

# 对数据进行权重调整
weighted_data = pd.DataFrame()
for major in major_versions:
    segment = data[(data['version number'] >= major) &
                   (data['version number'] < major + 1)]
    if major in latest_three_majors.values:
        # 对最新的三个大版本增加 10% 的数据点
        extra_data = segment.sample(frac=0.1, replace=True)
        weighted_segment = pd.concat([segment, extra_data])
    else:
        # 对其他版本减少 5% 的数据点
        reduced_data = segment.sample(frac=0.95, replace=False)
        weighted_segment = reduced_data
    weighted_data = pd.concat([weighted_data, weighted_segment])

# 使用调整后的数据重新进行线性回归
segment_models = {}
segment_data = {}
for major in major_versions:
    segment = weighted_data[(weighted_data['version number'] >= major) &
                            (weighted_data['version number'] < major + 1)]
    X = segment[['version number']]
    y = segment['days_since_first_release']
    model = LinearRegression().fit(X, y)
    segment_models[major] = model
    segment_data[major] = segment


# 初始化线性回归模型的字典来存储每个大版本段的模型
segment_models = {}
segment_data = {}

# 对每个大版本的小版本序列进行线性回归
for i in range(len(major_versions)):
    # 获取当前大版本的数据段
    major = major_versions.iloc[i]
    if i + 1 < len(major_versions):
        next_major = major_versions.iloc[i + 1]
    else:
        next_major = data['version number'].max() + 1

    segment = data[(data['version number'] >= major) & (data['version number'] < next_major)]
    X = segment[['version number']]
    y = segment['days_since_first_release']
    model = LinearRegression().fit(X, y)
    segment_models[major] = model
    segment_data[major] = segment

# 绘制原始数据和每个段的拟合线

plt.figure(figsize=(10, 6))
plt.scatter(data['version number'], data['Date'], label='实际发布日期', color='black')

for major, model in segment_models.items():
    seg = segment_data[major]
    X_test = pd.DataFrame(np.linspace(seg['version number'].min(), seg['version number'].max(), 100), columns=['version number'])
    y_pred = model.predict(X_test)
    plt.plot(X_test, [data['Date'].min() + timedelta(days=int(day)) for day in y_pred], label=f'版本 {major} 拟合线')

plt.xlabel('版本号')
plt.ylabel('发布日期')
plt.title('iOS 版本发布日期与分段线性回归拟合')
plt.legend()
plt.show()


def predict_release_date(version_number):
    # 将版本号转换为浮点数
    version_number = float(version_number)
    # 确定版本号所属的大版本段
    for start_version in sorted(segment_models.keys(), reverse=True):
        if version_number >= start_version:
            model = segment_models[start_version]
            # 使用相应的模型进行预测，注意这里传入 DataFrame 并指定列名
            predict_data = pd.DataFrame([[version_number]], columns=['version number'])
            predicted_days = model.predict(predict_data)[0]
            # 将预测的天数转换为日期
            predicted_date = data['Date'].min() + pd.Timedelta(days=predicted_days)
            return predicted_date

# 从用户那里获取要预测的版本号
version_to_predict = input("请输入您想要预测的版本号：")

# 预测并打印发布日期
predicted_date = predict_release_date(version_to_predict)
print(f"预测的版本 {version_to_predict} 的发布日期为：{predicted_date.strftime('%Y-%m-%d')}")
