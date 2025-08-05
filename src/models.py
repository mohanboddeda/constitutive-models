import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=6,output_dim=6, d_model=128, nhead=4, num_layers=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, output_dim)
                )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.mlp_head(x)
    
    
class MLP(nn.Module):
    def __init__(self, input_dim=([3,3]), output_dim=6, hidden_dim=128, num_layers=3, dropout=0.1, activation='relu'):
        super().__init__()
        layers = []
        in_dim = input_dim[0] * input_dim[1]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.mlp(x)