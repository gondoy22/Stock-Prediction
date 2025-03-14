from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib
import os


def predictions(companies, date_start):
    cache_key = tuple(companies) + (date_start,)
    if cache_key in temp_cache:
        return temp_cache[cache_key]

    loaded_models = {}
    predictions_dict = {}

    for name in companies:
        individual_cache_key = (name, date_start)
        model_path = os.path.join("models", f"{name}_model.pkl")
        if individual_cache_key in temp1_cache:
            _, _, X_test_array, scaled_y_test, _, scaler_y, _, _ = temp1_cache[individual_cache_key]


            if name not in loaded_models: 
                loaded_models[name] = joblib.load(model_path)
                print(f"Loaded model for {name}")

            test_generator = TimeseriesGenerator(X_test_array, scaled_y_test, length=50, batch_size=1)
            predictions_test = loaded_models[name].predict(test_generator, verbose=0) 
            predictions_original_test = scaler_y.inverse_transform(predictions_test).ravel()
            predictions_dict[name] = predictions_original_test

        else:
            loaded_models[name] = joblib.load(model_path)
            print(f"Loaded model for {name}")

            _, _, X_test_array, scaled_y_test, _, scaler_y, _, _ = datasets(name, date_start)
            test_generator = TimeseriesGenerator(X_test_array, scaled_y_test, length=50, batch_size=1)
            predictions_test = loaded_models[name].predict(test_generator, verbose=0)
            predictions_original_test = scaler_y.inverse_transform(predictions_test).ravel()
            predictions_dict[name] = predictions_original_test

    temp_cache[cache_key] = predictions_dict
    return predictions_dict