# Example of creating and training a simple model on GPU
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Check GPU availability
if tf.config.experimental.list_physical_devices('GPU'):
    # Define and compile a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(784,), activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generate some random data for training
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784) / 255.0

    # Train the model on GPU
    model.fit(x_train, y_train, epochs=5)
