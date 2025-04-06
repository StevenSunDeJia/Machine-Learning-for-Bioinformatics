## Defining a Graph-Embedded Neural Network Model Based on Kong & Yu (2018)

import tensorflow as tf
from tensorflow import keras
from keras import layers

### Custom graph-embedded Dense layer (for reference)

class GraphEmbeddedDense(layers.Layer):
    def __init__(self, units, mask, use_bias=True, **kwargs):

        """
        Custom Dense layer that embeds a feature graph.
        
        Args:
            units (int): Number of output units (typically equals the input dimension).
            mask (np.array or tf.Tensor): Adjacency matrix of shape (input_dim, units).
            use_bias (bool): Whether to include a bias term.
        """

        super(GraphEmbeddedDense, self).__init__(**kwargs)
        self.units = units
        self.mask = tf.constant(mask, dtype=tf.float32)
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.HeUniform(),
            trainable=True,
            name="kernel"
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias"
            )
        else:
            self.bias = None
        super(GraphEmbeddedDense, self).build(input_shape)

    def call(self, inputs):
        masked_kernel = self.kernel * self.mask
        output = tf.matmul(inputs, masked_kernel)
        if self.use_bias:
            output = output + self.bias
        return output

### Build a simple Keras model that uses the custom graph-embedded layer.

def create_graph_model(input_dim, hidden_dims, mask):
    inputs = keras.Input(shape=(input_dim,))
    # Graph-embedded layer: name it "graph_layer"
    x = GraphEmbeddedDense(units=input_dim, mask=mask, name="graph_layer")(inputs)
    x = layers.ReLU()(x)
    # First fully connected layer, name it "fc1"
    x = layers.Dense(hidden_dims[0], activation="relu", name="fc1")(x)
    # Additional Dense layers (if any)
    for hdim in hidden_dims[1:]:
        x = layers.Dense(hdim, activation="relu")(x)
    # Final output layer for binary classification
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

### Evaluate Feature Importance Using the GCW Method

def evaluate_feature_importance_keras(model):

    """
    Evaluate feature importance using the GCW method from Kong & Yu (2018).
    
    Assumptions:
      - The graph-embedded layer is named "graph_layer". It has attributes:
            • kernel: trainable weight matrix of shape (p, p)
            • mask: fixed binary tensor (adjacency matrix) of shape (p, p)
      - The first Dense layer following the graph layer is named "fc1" 
        and has kernel shape (p, h1), where rows correspond to input features.
    
    Returns:
        importance: a tensor of shape (p,) with the computed importance scores.
    """

    # Retrieve the graph-embedded layer
    graph_layer = model.get_layer("graph_layer")
    # Retrieve the trainable kernel and the fixed mask
    W_in = graph_layer.kernel  # shape: (p, p)
    mask = graph_layer.mask    # shape: (p, p) assumed to be a tf.Tensor
    
    # Compute the effective weight matrix (only allowed connections)
    effective_W = W_in * mask  # element-wise multiplication
    
    # For each feature j, sum the absolute weights from the graph layer.
    # Since Keras Dense layers perform: output = x @ kernel, the j-th column
    # corresponds to contributions from feature j.
    graph_contrib = tf.reduce_sum(tf.abs(effective_W), axis=0)  # shape: (p,)
    
    # Retrieve the first Dense layer after the graph layer (named "fc1")
    fc1 = model.get_layer("fc1")
    # For a Dense layer, kernel shape is (input_dim, units); rows correspond to input features.
    W_fc1 = fc1.kernel  # shape: (p, h1)
    fc1_contrib = tf.reduce_sum(tf.abs(W_fc1), axis=1)  # sum over units → shape: (p,)
    
    # Compute the degree of each feature from the mask (sum over each column)
    degree = tf.reduce_sum(mask, axis=0)  # shape: (p,)
    eps = 1e-8  # to avoid division by zero
    c = 50
    gamma = tf.minimum(tf.ones_like(degree), c / (degree + eps))
    
    # Combine the contributions with the penalty factor
    importance = gamma * graph_contrib + fc1_contrib
    return importance

## Defining a Baseline Model

def create_baseline_model(input_dim, hidden_dims):

    """
    Creates a baseline deep feedforward network with the same architecture
    as the graph-embedded model but without domain-specific graph information.
    
    Architecture:
      - Input layer of dimension 'input_dim'.
      - First hidden layer: Fully connected mapping from input to an output 
        with the same dimension (i.e. input_dim), with ReLU activation.
      - Additional Dense hidden layers as specified in hidden_dims.
      - Final output layer with a single neuron for binary classification.
    
    Args:
      input_dim (int): Number of input features.
      hidden_dims (list of int): List of hidden layer sizes after the first layer.
    
    Returns:
      model (tf.keras.Model): The baseline Keras model.
    """

    inputs = keras.Input(shape=(input_dim,))
    
    # First hidden layer: fully connected (without graph-based filtering)
    x = layers.Dense(input_dim, activation="relu", name="baseline_fc1")(inputs)
    
    # Additional hidden layers
    for i, hdim in enumerate(hidden_dims):
        x = layers.Dense(hdim, activation="relu", name=f"baseline_fc{i+2}")(x)
    
    # Final output layer (for binary classification)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
