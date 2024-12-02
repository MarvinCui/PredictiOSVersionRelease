import pandas as pd
from sklearn.linear_model import LinearRegression
import json

# 数据加载和预处理（保持你的逻辑）
data = pd.read_excel('Resources/iOS release date.xlsx')
data['days_since_first_release'] = (data['Date'] - data['Date'].min()).dt.days

data['is_major'] = data['version number'].apply(lambda x: x.is_integer())
major_versions = data[data['is_major']]['version number']
latest_three_majors = major_versions.sort_values(ascending=False).head(3)

weighted_data = pd.DataFrame()
for major in major_versions:
    segment = data[(data['version number'] >= major) &
                   (data['version number'] < major + 1)]
    if major in latest_three_majors.values:
        extra_data = segment.sample(frac=0.1, replace=True)
        weighted_segment = pd.concat([segment, extra_data])
    else:
        reduced_data = segment.sample(frac=0.95, replace=False)
        weighted_segment = reduced_data
    weighted_data = pd.concat([weighted_data, weighted_segment])

# 模型训练并保存结果
models = {}
for major in major_versions:
    segment = weighted_data[(weighted_data['version number'] >= major) &
                            (weighted_data['version number'] < major + 1)]
    X = segment[['version number']]
    y = segment['days_since_first_release']
    model = LinearRegression().fit(X, y)
    models[major] = {
        "coef": model.coef_[0],
        "intercept": model.intercept_
    }

# 保存模型和基本数据
output = {
    "models": models,
    "min_date": data['Date'].min().strftime('%Y-%m-%d')
}
with open('models.json', 'w') as f:
    json.dump(output, f)