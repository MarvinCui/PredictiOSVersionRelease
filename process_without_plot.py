import pandas as pd
from sklearn.linear_model import LinearRegression

# 加载数据
data = pd.read_excel('iOS release date.xlsx')
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
version_to_predict = input("Please enter the version number you want to predict: ")

# 预测并打印发布日期
predicted_date = predict_release_date(version_to_predict)
print(f"The predict version {version_to_predict} may released in {predicted_date.strftime('%Y-%m-%d')}")