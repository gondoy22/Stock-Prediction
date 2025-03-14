from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


def model_building(companies, date_start):
    #Saving models in models directory in pkl file. They will be used later
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    for name in companies:
        #Calling test and train sets prepared in Data Exctraction set.
        X_train_array, scaled_y_train,X_test_array,scaled_y_test,scaled_merged_train,_,_,_ = datasets(name, date_start)
        train_generator = TimeseriesGenerator(X_train_array,scaled_y_train, length = 50, batch_size =1)
        test_generator = TimeseriesGenerator(X_test_array,scaled_y_test, length = 50, batch_size =1)

        #Building LSTM model
        model = Sequential()

        model.add(keras.Input(shape = (50,scaled_merged_train.shape[-1])))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer = Adam(learning_rate=0.001), loss = 'mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(train_generator, epochs=100, validation_data=test_generator, callbacks=[early_stopping])
        model_path = os.path.join(model_dir, f"{name}_model.pkl")
        joblib.dump(model, model_path)


    return
