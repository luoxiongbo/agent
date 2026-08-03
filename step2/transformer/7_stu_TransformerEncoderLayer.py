from torch import nn

encoder_layer = nn.TransformerEncoderLayer(16, 4)

decoder_layer = nn.TransformerDecoderLayer(16, 4)

class TinyTransformer(nn.Module):

