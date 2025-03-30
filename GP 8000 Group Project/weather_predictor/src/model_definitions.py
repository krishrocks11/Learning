import torch
import torch.nn as nn

# Define the transformer-based model with dropout and batch_first=True.
class WeatherTransformer(nn.Module):
    """
    Transformer-based model for weather prediction using cyclic features.
    """
    def __init__(self, emb_dim=32, nhead=4, num_layers=2, dropout=0.1, output_dim=3):
        """
        Args:
            emb_dim (int): Embedding dimension. Must be divisible by nhead.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout rate.
            output_dim (int): Number of output targets (e.g., max temp, min temp, wind speed).
        """
        super(WeatherTransformer, self).__init__()
        if emb_dim % nhead != 0:
             # Adjust emb_dim to be divisible by nhead if needed, or raise error
             # For simplicity, let's raise an error here. User should ensure this via hyperparameter tuning.
             raise ValueError(f"emb_dim ({emb_dim}) must be divisible by nhead ({nhead})")

        # Input projection: from scalar feature per time step to embedding vector.
        self.input_proj = nn.Linear(1, emb_dim) # Input is (batch, seq_len=6, 1)

        # Transformer encoder layer with dropout. batch_first=True expects (batch, seq_len, features).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Optional dropout before the final Fully Connected layer.
        self.dropout = nn.Dropout(dropout)
        # Final layer to map to the desired output dimension.
        self.fc = nn.Linear(emb_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len=6, 1).
        Returns:
            torch.Tensor: Output tensor of shape (batch, output_dim).
        """
        # x shape: (batch, seq_len=6, 1)
        x = self.input_proj(x)  # Project to embedding: (batch, seq_len=6, emb_dim)
        x = self.transformer_encoder(x)  # Pass through transformer: (batch, seq_len=6, emb_dim)

        # Aggregate features across the sequence dimension. Mean pooling is a common choice.
        x = x.mean(dim=1)  # Shape: (batch, emb_dim)

        x = self.dropout(x) # Apply dropout
        output = self.fc(x)  # Final prediction: (batch, output_dim)
        return output


# Define the MLP model.
class MLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) model for weather prediction.
    """
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.1, output_dim=3):
        """
        Args:
            input_dim (int): Number of input features (typically 6 cyclic features).
            hidden_dim (int): Size of hidden layers.
            num_layers (int): Number of hidden layers.
            dropout (float): Dropout rate.
            output_dim (int): Number of output targets.
        """
        super(MLP, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Final output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the MLP.
        Args:
            x (torch.Tensor): Input tensor. Expected shape (batch, seq_len=6, 1)
                              from WeatherDataset, needs flattening.
        Returns:
            torch.Tensor: Output tensor of shape (batch, output_dim).
        """
        # x comes with shape (batch, seq_len, 1); flatten it to (batch, input_dim)
        # Ensure the flattening matches the expected input_dim (usually 6).
        if x.shape[1] * x.shape[2] != 6:
             # This check assumes input_dim is always 6 for this model.
             # Adjust if input_dim can vary.
             raise ValueError(f"Input tensor shape {x.shape} does not flatten to the expected input dimension 6.")
        x = x.view(x.size(0), -1) # Flatten to (batch, 6)
        return self.net(x)
