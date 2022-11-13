import torch as T
import torch.nn as nn

input_size = 10
hidden_size = 128
num_layers = 3
seq_len = 23
batch_first = True
num_directions = 2  # or 2 for bidir
batch_size = 7

rnn = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=batch_first,
    bidirectional=(num_directions == 2),
)

if batch_first:
    x = T.randn(batch_size, seq_len, input_size)
else:
    x = T.randn(seq_len, batch_size, input_size)

h0 = T.randn(num_layers * num_directions, batch_size, hidden_size)
c0 = T.randn(num_layers * num_directions, batch_size, hidden_size)


"""
output contains the concatenation of all hidden states at each timestep
if batch_first => output = (batch_size, seq_len, num_directions * hidden_size)
else           => output = (seq_len, batch_size, num_directions * hidden_size)
last timestep
(hn, cn) => (num_directions * num_layers, batch_size, hidden_size)
"""
output, (hn, cn) = rnn(x, (h0, c0))
