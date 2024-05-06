import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data():
    X_and_op = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y_and_op = np.asarray([[0], [0], [0], [1]], dtype=np.float32)

    X_or_op = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y_or_op = np.asarray([[0], [1], [1], [1]], dtype=np.float32)

    X_xor_op = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y_xor_op = np.asarray([[0], [1], [1], [0]], dtype=np.float32)

    return (X_and_op, Y_and_op), (X_or_op, Y_or_op), (X_xor_op, Y_xor_op)

def create_model(X, Y, activation_func, epochs=1000):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, input_dim=2, activation=activation_func),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, Y, epochs=epochs, verbose=0)
    return model



def evaluate_model(model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

(X_and, Y_and), (X_or, Y_or), (X_xor, Y_xor) = prepare_data()

and_model = create_model(X_and, Y_and, activation_func='relu')
or_model = create_model(X_or, Y_or, activation_func='relu')
xor_model = create_model(X_xor, Y_xor, activation_func='relu')

print("Testing AND Model:")
evaluate_model(and_model, X_and, Y_and)
print("\nTesting OR Model:")
evaluate_model(or_model, X_or, Y_or)
print("\nTesting XOR Model:")
evaluate_model(xor_model, X_xor, Y_xor)