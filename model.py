from tensorflow.python import keras
import json

def model():
    # Define the model using Keras.
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model_json = model.to_json()
    # print("path : ",data_path)
    # base = Path('/data/')
    # base.mkdir(exist_ok=True)
    # with open(f"{data_path}/model.json", "w") as json_file:
        # json_file.write(model_json)
    return model

if __name__ == "__main__":
    print(model().summary())