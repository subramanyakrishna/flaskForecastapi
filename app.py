from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request


forecastingModel = load_model('forecasting_model')


def getReverseMinMaxvalue(minMaxedValue, min, max):
    return minMaxedValue * (max-min) + min


scaler = MinMaxScaler(feature_range=(0, 1))

app = Flask(__name__)


@app.route('/getForecast', methods=['POST'])
def get_forcast():
    request_body = request.get_json()
    tenDayaOutputFromApi = request_body['tenDaysOutputs']
    max_value = max(tenDayaOutputFromApi)
    min_value = min(tenDayaOutputFromApi)
    transformedInputs = scaler.fit_transform(
        np.array(tenDayaOutputFromApi).reshape(-1, 1))
    tenDaySolarOutput = transformedInputs.reshape(1, 10, 1)
    next3DaysOutput = []
    for i in range(0, 3):
        pre = forecastingModel.predict(tenDaySolarOutput)
        # 8th day is added for next prediction
        tenDaySolarOutput = np.append(tenDaySolarOutput, pre[0][0])
        tenDaySolarOutput = tenDaySolarOutput[1:]  # first day is removed
        tenDaySolarOutput = tenDaySolarOutput.reshape(1, 10, 1)

        next3DaysOutput.append(tenDaySolarOutput[0][9][0])
    next3DaysOutput = [round(getReverseMinMaxvalue(
        dayOutput, min_value, max_value), 2) for dayOutput in next3DaysOutput]

    return {'next3DaysOutput': next3DaysOutput}


if(__name__ == "__main__"):
    app.run(debug=True)
