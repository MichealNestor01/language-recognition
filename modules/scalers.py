from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scaleStandard(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def scaleMinMax(features):
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)
