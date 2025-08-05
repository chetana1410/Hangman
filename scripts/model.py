import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.optim.adam import Adam

from scripts.vocabulary import Vocabulary

class EmbeddingLayer(nn.Module):
    """A PyTorch module for word embeddings."""
    
    def __init__(self, embedding_dim: int, vocab_size: int):
        """
        Initialize the embedding layer.

        Args:
            embedding_dim (int): The dimensionality of the embedding vectors.
            vocab_size (int): The size of the vocabulary.
        """
        super(EmbeddingLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer.

        Args:
            input_ids (torch.Tensor): Input tensor containing the indices of the letters to be embedded.

        Returns:
            torch.Tensor: Embedded input tensor.
        """
        embedded_input = self.embedding_table(input_ids)
        return embedded_input


class PositionalEncoder(nn.Module):
    """A PyTorch module for computing positional encodings."""
    
    def __init__(self, embedding_dim: int, dropout_prob: float, max_sequence_length: int = 5000):
        """
        Initialize the positional encoder.

        Args:
            embedding_dim (int): The dimensionality of the embedding vectors.
            dropout_prob (float): The dropout probability.
            max_sequence_length (int, optional): The maximum sequence length. Defaults to 5000.
        """
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

        # Compute the positional encodings once in log space.
        self.positional_encodings = self._compute_positional_encodings()

    def _compute_positional_encodings(self) -> torch.Tensor:
        """
        Compute the positional encodings.

        Returns:
            torch.Tensor: The positional encodings tensor.
        """
        pe = torch.zeros(self.max_sequence_length, self.embedding_dim, requires_grad=False)
        positions = torch.arange(0, self.max_sequence_length).unsqueeze(1)
        freqs = torch.exp(torch.arange(0, self.embedding_dim, 2) * -(math.log(10000.0) / self.embedding_dim))
        pe[:, 0::2] = torch.sin(positions * freqs)
        pe[:, 1::2] = torch.cos(positions * freqs)
        return pe.unsqueeze(0)  # Shape: (1, max_sequence_length, embedding_dim)

    def forward(self, input_tensor: torch.Tensor, separator_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoder.

        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length, embedding_dim).
            separator_indices (torch.Tensor): A tensor of shape (batch_size) containing the indices of the '&' separator.

        Returns:
            torch.Tensor: The output tensor with positional encodings added to tokens before the separator.
        """
        # Get positional encodings for the sequence length
        positional_encodings = self.positional_encodings[:, :input_tensor.size(1), :].to(input_tensor.device)
        batch_size, seq_length, _ = input_tensor.shape
        
        # Create a mask to apply positional encodings up to the separator index
        mask = torch.arange(seq_length, device=input_tensor.device).unsqueeze(0) < separator_indices.unsqueeze(1)
        mask = mask.unsqueeze(2).expand(-1, -1, self.embedding_dim)
        
        # Apply positional encoding where mask is True
        input_tensor = input_tensor + positional_encodings * mask
        return self.dropout(input_tensor)


class SegmentEmbeddingLayer(nn.Module):
    """A PyTorch module for segment embeddings."""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize the segment embedding layer.

        Args:
            embedding_dim (int): The dimensionality of the embedding vectors.
        """
        super(SegmentEmbeddingLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.segment_embeddings = nn.Embedding(2, embedding_dim)  # Embedding for segment 0 and 1

    def forward(self, input_ids: torch.Tensor, sep_token_id: int):
        """
        Forward pass of the segment embedding layer.

        Args:
            input_ids (torch.Tensor): Input tensor containing the indices of the letter to be embedded.
            sep_token_id (int): Seperator token id corresponding to '&'
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Segment embeddings tensor
                - torch.Tensor: Indices of '&' token for each vector (shape: [batch_size]).
        """
        # Find the indices of the separator token
        sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=False)

        # Create segment IDs where 0 is for tokens before the separator, 1 after
        segment_ids = torch.zeros_like(input_ids, dtype=torch.long)
        sep_mask = (input_ids == sep_token_id).cumsum(dim=1)
        segment_ids[sep_mask >= 1] = 1  # Set tokens after the first separator as 1

        # Get segment embeddings
        segment_embeddings = self.segment_embeddings(segment_ids)
        return segment_embeddings, sep_indices[:, 1]


class CombinedEmbedding(nn.Module):
    """
    A module that combines word embeddings, segment embeddings, and positional encodings for sequence data.

    Args:
        vocab_size (int): The size of the vocabulary.
        model_dim (int): The dimension of the embedding vectors.
        dropout_prob (float): The probability of dropping out some elements in the embedding layer.
        sep_token_id (int): The token id that represents the separator between segments.
    """

    def __init__(self, vocab_size: int, model_dim: int, dropout_prob: float, sep_token_id: int):
        """
        Initializes a CombinedEmbedding instance.
    
        This class combines word embeddings, segment embeddings, and positional encodings to create an embedding layer
        suitable for transformer models. It takes in the size of the vocabulary, the dimension of the model, the dropout
        probability, and the ID of the separator token.
    
        Args:
            vocab_size (int): The number of unique words in the vocabulary.
            model_dim (int): The dimensionality of the model embeddings.
            dropout_prob (float): The probability of dropping out elements during training to prevent overfitting.
            sep_token_id (int): The ID of the separator token used to differentiate between different segments in the input.
    
        Attributes:
            embedding_layer (EmbeddingLayer): An instance of EmbeddingLayer for word embeddings.
            segment_embedding_layer (SegmentEmbeddingLayer): An instance of SegmentEmbeddingLayer for segment embeddings.
            positional_encoder (PositionalEncoder): An instance of PositionalEncoder to add positional encodings.
            sep_token_id (int): The ID of the separator token used in the input.
            embedding_dim (int): The dimension of the model embeddings, which is also the dimension of the positional encodings.
        """
        super(CombinedEmbedding, self).__init__()
        self.embedding_layer = EmbeddingLayer(embedding_dim=model_dim, vocab_size=vocab_size)
        self.segment_embedding_layer = SegmentEmbeddingLayer(embedding_dim=model_dim)
        self.positional_encoder = PositionalEncoder(embedding_dim=model_dim, dropout_prob=dropout_prob)
        self.sep_token_id = sep_token_id
        self.embedding_dim = model_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CombinedEmbedding module.

        Args:
            input_ids (torch.Tensor): The tokenized inputs with shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: The combined embeddings after adding word and segment embeddings along with positional encodings.
        """
        # Get normal word embeddings
        word_embeddings = self.embedding_layer(input_ids)
        # Get segment embeddings (0 for tokens before '&', 1 for after)
        segment_embeddings, sep_indices = self.segment_embedding_layer(input_ids, self.sep_token_id)
        # Add word embeddings and segment embeddings together
        combined_embeddings = (word_embeddings + segment_embeddings) / math.sqrt(self.embedding_dim)
        # Apply positional encoding before the seperator token
        return self.positional_encoder(combined_embeddings, sep_indices)


class FeedForwardNetwork(nn.Module):
    """
    A feedforward neural network with GELU activation, commonly used in transformer models.

    The architecture is as follows:
    FFN(x) = max(0, xW1 + b1)W2 + b2

    This consists of two linear layers with a GELU activation in between and dropout for regularization.

    Args:
    - model_dim (int): The input and output dimension of the model (typically the embedding dimension).
    - ff_hidden_dim (int): The hidden layer dimension in the feedforward network.
    - dropout_rate (float): The dropout rate applied after the first linear layer (default is 0.1).
    """
    
    def __init__(self, model_dim, ff_hidden_dim, dropout_rate=0.1):
        """
        Initializes a FeedForwardNetwork instance.
    
        This class represents a feed-forward network used within transformer models. It consists of two linear layers followed by a dropout layer for regularization. The network processes the input by transforming it through these layers and
    applying dropout to prevent overfitting.
    
        Args:
            model_dim (int): The dimensionality of the input embeddings that will be processed by this feed-forward network.
            ff_hidden_dim (int): The hidden dimension size for the first linear transformation layer. This is typically larger than the `model_dim` to allow more complex transformations.
            dropout_rate (float, optional): The rate at which neurons are dropped during training to prevent overfitting. Defaults to 0.1.
    
        Attributes:
            fc1 (nn.Linear): First linear transformation layer with input dimension `model_dim` and output dimension `ff_hidden_dim`.
            fc2 (nn.Linear): Second linear transformation layer with input dimension `ff_hidden_dim` and output dimension `model_dim`.
            dropout (nn.Dropout): Dropout layer to apply regularization during training.
        """
        super(FeedForwardNetwork, self).__init__()
        
        # Linear layers
        self.fc1 = nn.Linear(model_dim, ff_hidden_dim)  # First linear transformation
        self.fc2 = nn.Linear(ff_hidden_dim, model_dim)  # Second linear transformation
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of the feedforward network.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, seq_length, model_dim).
        
        Returns:
        - output (Tensor): Output tensor of shape (batch_size, seq_length, model_dim).
        """
        # Apply first linear transformation followed by GELU activation, then dropout, and finally the second linear transformation
        x = self.fc1(x)               # (batch_size, seq_length, ff_hidden_dim)
        x = F.gelu(x)                 # Apply GELU activation
        x = self.dropout(x)           # Apply dropout
        output = self.fc2(x)          # (batch_size, seq_length, model_dim)
        return output


def scaled_dot_product_attention(query, key, value, mask=None, dropout_layer=None):
    """
    Scaled dot-product attention mechanism.

    Parameters:
    - query: Tensor of shape (batch_size, num_queries, d_k)
    - key: Tensor of shape (batch_size, num_keys, d_k)
    - value: Tensor of shape (batch_size, num_keys, d_v)
    - mask: Optional binary mask to indicate which positions to attend to.
    - dropout_layer: Optional dropout layer for regularization.

    Returns:
    - attention_output: Tensor of shape (batch_size, num_queries, d_v)
    - attention_weights: Tensor of shape (batch_size, num_queries, num_keys)
    """
    d_k = query.size(-1)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -np.inf)

    attention_weights = F.softmax(attention_scores, dim=-1)

    if dropout_layer is not None:
        attention_weights = dropout_layer(attention_weights)

    attention_output = torch.matmul(attention_weights, value)
    return attention_output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi Headed Attention layer of a Transformer Architecture"""
    
    def __init__(self, num_heads, model_dim, key_dim, value_dim, dropout_prob):
        """
        Multi-head attention mechanism.

        Parameters:
        - num_heads: Number of attention heads.
        - model_dim: Dimensionality of the model (embedding size).
        - key_dim: Dimensionality of the keys/queries.
        - value_dim: Dimensionality of the values.
        - dropout_prob: Dropout probability for regularization.
        """
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "Model dimension must be divisible by the number of heads."

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Linear layers to project inputs for multi-head attention
        self.query_proj = nn.Linear(key_dim * num_heads, model_dim, bias=False)
        self.key_proj = nn.Linear(key_dim * num_heads, model_dim, bias=False)
        self.value_proj = nn.Linear(value_dim * num_heads, model_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout_prob)

        # Output projection layer
        self.output_proj = nn.Linear(model_dim, num_heads * value_dim)

        self.attention_weights = None  # For visualization purposes, if needed

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the multi-head attention mechanism.

        Parameters:
        - query: Tensor of shape (batch_size, seq_length, embedding_size)
        - key: Tensor of shape (batch_size, seq_length, embedding_size)
        - value: Tensor of shape (batch_size, seq_length, embedding_size)
        - mask: Optional binary mask to prevent attending to certain positions.

        Returns:
        - output: Tensor of shape (batch_size, seq_length, embedding_size)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        seq_length = query.size(1)
        head_dim = self.model_dim // self.num_heads

        # Project and reshape the input tensors for multiple heads
        query = self.query_proj(query).view(batch_size, seq_length, self.num_heads, head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, seq_length, self.num_heads, head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, seq_length, self.num_heads, head_dim).transpose(1, 2)

        # Apply scaled dot-product attention
        attention_output, self.attention_weights = scaled_dot_product_attention(query, key, value, mask, self.dropout)

        # Concatenate attention output from all heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.num_heads * head_dim)

        # Apply output projection
        return self.output_proj(attention_output)


class Batch:
    """Custom defined Batch Class"""
    
    def __init__(self, source_seq, target_seq, guessed_seq, mask_token=None, pad_token=0):
        """
        Class to handle source and target sequences for a batch.

        Parameters:
        - source_seq: Tensor of source sequences (e.g., input sentences).
        - target_seq: Tensor of target sequences (e.g., output sentences).
        - mask_token: Token value used for masking elements in the sequence.
        - pad_token: Token value used for padding elements in the sequence (default is 0).
        """
        self.source_seq = source_seq
        self.target_seq = target_seq
        self.guessed_seq = guessed_seq
        
        # Create source mask where positions that are not masked or padded are set to True
        self.source_mask = ((source_seq != mask_token) & (source_seq != pad_token)).unsqueeze(-2)
        
        # Random mask to identify positions in the source sequence that are not mask tokens
        self.random_mask = (source_seq != mask_token)


class LayerNormalization(nn.Module):
    def __init__(self, num_features, epsilon=1e-6):
        """
        Layer normalization module.

        Parameters:
        - num_features: Number of features in the input tensor.
        - epsilon: A small constant to prevent division by zero (default is 1e-6).
        """
        super(LayerNormalization, self).__init__()
        
        # Learnable parameters for scaling and shifting
        self.scale_param = nn.Parameter(torch.ones(num_features))  # Equivalent to gamma in normalization
        self.shift_param = nn.Parameter(torch.zeros(num_features))  # Equivalent to beta in normalization
        self.epsilon = epsilon

    def forward(self, input_tensor):
        """
        Forward pass for layer normalization.

        Parameters:
        - input_tensor: Tensor of shape (batch_size, ..., num_features)

        Returns:
        - Normalized tensor with the same shape as input_tensor.
        """
        mean = input_tensor.mean(-1, keepdim=True)
        std_dev = input_tensor.std(-1, keepdim=True)
        
        # Normalize the input and apply learned scale and shift parameters
        return self.scale_param * (input_tensor - mean) / (std_dev + self.epsilon) + self.shift_param


class ComputeLossWithMask:
    """
    A class to compute a combined loss function using cross-entropy (CE) and cosine similarity (CosSim) losses, with optional masking applied during training.

    Attributes:
        model_output_generator (callable): A callable that takes `predicted_output` and `mask` as inputs and returns processed outputs and unmasked outputs.
        ce_loss_weight (float): Weight for the cross-entropy loss. Defaults to 1.0.
        cos_sim_weight (float): Weight for the cosine similarity loss. Defaults to 1.0.
        optimizer (Optimizer or None): An optional optimizer used for updating model parameters based on computed losses.

    Methods:
        __init__(self, model_output_generator, ce_loss_weight=1.0, cos_sim_weight=1.0, optimizer=None): Initializes the ComputeLossWithMask instance with given weights and optional optimizer.

        __call__(self, predicted_output, target_output, mask, guess_mask):
            Computes the combined loss for a batch of data using cross-entropy and cosine similarity losses.
            Args:
                predicted_output (torch.Tensor): Model's output tensor before processing with shape (batch_size, num_classes).
                target_output (torch.LongTensor): True labels tensor of shape (batch_size,) or (batch_size, num_classes) depending on the model architecture.
                mask (torch.Tensor): Mask tensor to indicate which elements should be considered for loss computation.
                guess_mask (torch.Tensor): Tensor representing guessed outputs used in cosine similarity calculation.
            Returns:
                torch.Tensor: The raw combined loss data.
    """
    def __init__(self, model_output_generator, ce_loss_weight=1.0, cos_sim_weight=1.0, optimizer=None):
        """
        Initializes the ComputeLossWithMask instance with given weights and optional optimizer.

        Args:
            model_output_generator (callable): A callable that takes `predicted_output` and `mask` as inputs and returns processed outputs and unmasked outputs.
            ce_loss_weight (float, optional): Weight for the cross-entropy loss. Defaults to 1.0.
            cos_sim_weight (float, optional): Weight for the cosine similarity loss. Defaults to 1.0.
            optimizer (Optimizer or None, optional): An optional optimizer used for updating model parameters based on computed losses.
        """
        self.model_output_generator = model_output_generator
        self.ce_loss_weight = ce_loss_weight
        self.cos_sim_weight = cos_sim_weight
        self.optimizer = optimizer

        if self.optimizer:
            self.optimizer.optimizer.zero_grad()

    
    def __call__(self, predicted_output, target_output, mask, guess_mask):
        """
        Computes the combined loss for a batch of data using cross-entropy and cosine similarity losses.

        Args:
            predicted_output (torch.Tensor): Model's output tensor before processing with shape (batch_size, num_classes).
            target_output (torch.LongTensor): True labels tensor of shape (batch_size,) or (batch_size, num_classes) depending on the model architecture.
            mask (torch.Tensor): Mask tensor to indicate which elements should be considered for loss computation.
            guess_mask (torch.Tensor): Tensor representing guessed outputs used in cosine similarity calculation.

        Returns:
            torch.Tensor: The raw combined loss data.
        """
        batch_size = predicted_output.shape[0]
        predicted_output, unmasked_output = self.model_output_generator(predicted_output, mask)

        ce_loss = F.cross_entropy(predicted_output, target_output)
        cos_sim_loss = F.cosine_similarity(unmasked_output, guess_mask, dim=-1).mean()

        # Combine both losses using the weights
        combined_loss = (self.ce_loss_weight * ce_loss) + (self.cos_sim_weight * cos_sim_loss)
        normalized_loss = combined_loss / batch_size
        
        # Backpropagation
        normalized_loss.backward()
        
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()

        # Return the raw loss data
        return combined_loss.data


class LearningRateScheduler:
    """
    A custom optimizer wrapper that adjusts the learning rate based on the Noam scheme.
    
    The learning rate is adjusted during training using the formula:
    
    learning_rate = (model_size^(-0.5)) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    
    This results in the learning rate increasing linearly during the warmup phase and then decreasing
    proportionally to the inverse square root of the step number afterward.
    """

    def __init__(self, model_size, scaling_factor, warmup_steps, optimizer):
        """
        Initializes the learning rate scheduler.

        Parameters:
        - model_size: The size of the model (typically the embedding dimension).
        - scaling_factor: A multiplicative factor for the learning rate.
        - warmup_steps: The number of steps for the warmup phase.
        - optimizer: The optimizer whose learning rate will be adjusted.
        """
        self.optimizer = optimizer
        self.model_size = model_size
        self.scaling_factor = scaling_factor
        self.warmup_steps = warmup_steps

        self.current_step = 0
        self.current_rate = 0

    def step(self):
        """
        Updates the learning rate and steps the optimizer.
        This method should be called at each training step.
        """
        self.current_step += 1
        new_learning_rate = self.compute_learning_rate()
        
        # Update the learning rate in the optimizer's parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate
        
        self.current_rate = new_learning_rate
        self.optimizer.step()

    def compute_learning_rate(self, step=None):
        """
        Computes the current learning rate based on the Noam formula.

        Parameters:
        - step: The current step number. If not provided, defaults to the internal step count.

        Returns:
        - The computed learning rate.
        """
        if step is None:
            step = self.current_step
        
        # Calculate the learning rate according to the Noam formula
        return self.scaling_factor * (self.model_size ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))


class ResidualSublayerConnection(nn.Module):
    """
    Implements a residual connection followed by layer normalization.
    For simplicity in this implementation, the normalization is applied before the sublayer,
    rather than after it as in the original Transformer paper.
    """
    
    def __init__(self, layer_size, dropout_probability):
        """
        Initializes the ResidualSublayerConnection.
        
        Parameters:
        - layer_size: The size of the input/output representations.
        - dropout_probability: The probability of dropout to be applied after the sublayer.
        """
        super(ResidualSublayerConnection, self).__init__()
        self.dropout_layer = nn.Dropout(dropout_probability)
        self.layer_normalization = LayerNormalization(layer_size)

    def forward(self, input_tensor, sublayer_function):
        """
        Applies the sublayer with a residual connection and layer normalization.
        
        Parameters:
        - input_tensor: The input to the sublayer.
        - sublayer_function: The sublayer function (e.g., attention or feed-forward layer).
        
        Returns:
        - The output tensor after applying the sublayer with a residual connection and layer normalization.
        """
        # Apply layer normalization to the input, then pass through the sublayer and apply dropout.
        normalized_output = self.layer_normalization(input_tensor)
        sublayer_output = sublayer_function(normalized_output)
        
        # Add the sublayer output (with dropout) to the input (residual connection).
        return input_tensor + self.dropout_layer(sublayer_output)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer composed of self-attention, feed-forward, and residual connections with layer normalization.
    """

    def __init__(self, self_attention_module, feed_forward_module, hidden_size, dropout_probability):
        """
        Initializes the TransformerEncoderLayer.

        Parameters:
        - self_attention_module: The self-attention mechanism (e.g., multi-head attention).
        - feed_forward_module: The feed-forward network.
        - hidden_size: The size of the hidden layer (typically the embedding size).
        - dropout_probability: The dropout probability for regularization.
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # Two residual connections: one for self-attention and one for the feed-forward layer.
        self.residual_connections = nn.ModuleList([copy.deepcopy(ResidualSublayerConnection(hidden_size, dropout_probability)) for _ in range(2)])
        
        self.self_attention = self_attention_module
        self.feed_forward = feed_forward_module
        self.hidden_size = hidden_size

    def forward(self, input_tensor, attention_mask):
        """
        Applies the encoder layer: self-attention followed by a feed-forward layer, both with residual connections.
        
        Parameters:
        - input_tensor: The input to the encoder layer.
        - attention_mask: The attention mask to be applied during self-attention.

        Returns:
        - The output after applying self-attention and feed-forward with residual connections.
        """
        # Apply the first residual connection with self-attention
        attention_output = self.residual_connections[0](
            input_tensor, 
            lambda x: self.self_attention(x, x, x, attention_mask)
        )
        
        # Apply the second residual connection with the feed-forward network
        return self.residual_connections[1](attention_output, self.feed_forward)


class Encoder(nn.Module):
    """
    A complete Encoder model consisting of multiple transformer layers (encoders) with self-attention.
    """
    def __init__(self, encoder_layer: nn.Module, output_generator: nn.Module, embedding_layer: nn.Module, num_layers: int):
        """
        Initializes the Encoder model.

        :param encoder_layer: A transformer encoder layer utilizing self-attention.
        :param num_layers: The number of encoder layers in the model.
        :param output_generator: The generator for output prediction.
        :param embedding_layer: Embedding layer for input token embeddings.
        """
        super(Encoder, self).__init__()
        self.encoder_layer = encoder_layer
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.embedding_layer = embedding_layer
        self.layer_normalization = LayerNormalization(encoder_layer.hidden_size)
        self.output_generator = output_generator

    def forward(self, input_tokens: torch.Tensor, source_mask: torch.Tensor):
        """
        Forward pass for the Encoder model.

        :param input_tokens: A tensor of shape (batch_size, sequence_length) representing input token indices.
        :param source_mask: A tensor of shape (batch_size, 1, sequence_length) representing the source mask for attention.
        :return: A tensor of shape (batch_size, sequence_length, hidden_size) after passing through all encoder layers and layer normalization.
        """
        embedded_tokens = self.embedding_layer(input_tokens)
        for encoder in self.encoder_layers:
            embedded_tokens = encoder(embedded_tokens, source_mask)
        return self.layer_normalization(embedded_tokens)

    def device(self):
        """
        Returns the device on which the model's parameters are stored.
        """
        return self.output_generator.output_linear.weight.device


class SimpleGenerator(nn.Module):
    """
    The output generator that converts the transformer encoder's output into log-probabilities over the vocabulary.
    """
    def __init__(self, hidden_size: int, vocabulary_size: int):
        """
        Initializes the output generator.

        :param hidden_size: The size of the hidden representations from the encoder.
        :param vocabulary_size: The size of the output vocabulary.
        """
        super(SimpleGenerator, self).__init__()
        self.output_linear = nn.Linear(in_features=hidden_size, out_features=vocabulary_size)

    def forward(self, encoder_outputs: torch.Tensor, token_mask: torch.Tensor):
        """
        Forward pass for the output generator.

        :param encoder_outputs: A tensor of shape (batch_size, sequence_length, hidden_size) from the transformer encoder.
        :param token_mask: A tensor of shape (batch_size, sequence_length) representing tokens to mask in the output.
        :return: Log-softmax over the vocabulary of shape (batch_size, vocabulary_size).
        """
        max_logits, _ = torch.max(self.output_linear(encoder_outputs), dim=1)
        masked_logits = max_logits.masked_fill(token_mask == 1, -1e9)  # Masking the already guessed ones
        return F.log_softmax(masked_logits, dim=1), F.log_softmax(max_logits, dim=1)
