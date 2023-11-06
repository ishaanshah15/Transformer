
import pickle
import random
import os
from typing import List, Optional, Tuple, Dict

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import ticker
from collections import defaultdict

import torch
from torch.nn import Module, Linear, Softmax, ReLU, LayerNorm, ModuleList, Dropout, Embedding, CrossEntropyLoss
from torch.optim import Adam

class PositionalEncodingLayer(Module):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (batch_size, sequence_length, embedding_dim)

        This function should create the positional encoding matrix
        and return the sum of X and the encoding matrix.

        The positional encoding matrix is defined as follow:

        P_(pos, 2i) = sin(pos / (10000 ^ (2i / d)))
        P_(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d)))

        The output will have shape (batch_size, sequence_length, embedding_dim)
        """
        # TODO: Implement positional encoding
        seq_length = X.shape[1]
        embedding_dim = X.shape[2]
         

        num_indices = int(np.ceil(embedding_dim/2))

        denom = 10000**(2*torch.arange(num_indices)/embedding_dim)[None]

        numerator = torch.arange(seq_length,dtype=torch.float)[None].T

        z = torch.matmul(numerator,1/denom) #shape is (sequence length, np.ceil(embedding_size/2))

        z_rep = torch.repeat_interleave(z,repeats=2,dim=1)
    
        sin_pos = torch.sin(z_rep) #shape is (sequence length, 2*np.ceil(embedding_size/2))
        cos_pos = torch.cos(z_rep) #shape is (sequence length, 2*np.ceil(embedding_size/2))

        sin_mask = torch.cat((torch.ones(seq_length,1),torch.zeros(seq_length,1)),dim=1)
        sin_mask = sin_mask.repeat(1,num_indices)

        cos_mask = torch.cat((torch.zeros(seq_length,1),torch.ones(seq_length,1)),dim=1)
        cos_mask = cos_mask.repeat(1,num_indices)


        pos = sin_pos*sin_mask + cos_pos*cos_mask
        pos = pos[:,:embedding_dim][None]

        
        return X + pos



class SelfAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.linear_Q = Linear(in_dim, out_dim)
        self.linear_K = Linear(in_dim, out_dim)
        self.linear_V = Linear(in_dim, out_dim)

        self.softmax = Softmax(-1)

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query_X, key_X and value_X have shape (batch_size, sequence_length, in_dim). The sequence length
        may be different for query_X and key_X but must be the same for key_X and value_X.

        This function should return two things:
            - The output of the self-attention, which will have shape (batch_size, sequence_length, out_dim)
            - The attention weights, which will have shape (batch_size, query_sequence_length, key_sequence_length)

        If a mask is passed as input, you should mask the input to the softmax, using `float(-1e32)` instead of -infinity.
        The mask will be a tensor with 1's and 0's, where 0's represent entries that should be masked (set to -1e32).

        Hint: The following functions may be useful
            - torch.bmm (https://pytorch.org/docs/stable/generated/torch.bmm.html)
            - torch.Tensor.masked_fill (https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)
        """
        # TODO: Implement the self-attention layer

        Q = self.linear_Q(query_X) #(batch_sz,sequence_len_Q,out_dim)
        K = self.linear_K(key_X).permute(0,2,1) #(batch_sz,out_dim,sequence_len_K)
        V = self.linear_V(value_X) #(batch_sz,sequence_len_K,out_dim)

        

        att_mat = torch.bmm(Q,K)/(self.out_dim**0.5) #(batch_sz,sequence_len_Q,sequence_len_K)


        if mask != None:
            att_mat = att_mat.masked_fill(mask == 0,value=float(-1e32))

        att_mat = self.softmax(att_mat) #(batch_sz,sequence_len_Q,sequence_len_K)

        return torch.bmm(att_mat,V),att_mat

       

class MultiHeadedAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention_heads = ModuleList([SelfAttentionLayer(in_dim, out_dim) for _ in range(n_heads)])

        self.linear = Linear(n_heads * out_dim, out_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function calls the self-attention layer and returns the output of the multi-headed attention
        and the attention weights of each attention head.

        The attention_weights matrix has dimensions (batch_size, heads, query_sequence_length, key_sequence_length)
        """

        outputs, attention_weights = [], []

        for attention_head in self.attention_heads:
            out, attention = attention_head(query_X, key_X, value_X, mask)
            outputs.append(out)
            attention_weights.append(attention)

        outputs = torch.cat(outputs, dim=-1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return self.linear(outputs), attention_weights

class EncoderBlock(Module):

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

    def forward(self, X, mask=None):
        """
        Implementation of an encoder block. Both the input and output
        have shape (batch_size, source_sequence_length, embedding_dim).

        The mask is passed to the multi-headed self-attention layer,
        and is usually used for the padding in the encoder.
        """
        att_out, _ = self.attention(X, X, X, mask)

        residual = X + self.dropout1(att_out)

        X = self.norm1(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)

        residual = X + self.dropout2(temp)

        return self.norm2(residual)

class Encoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([EncoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])
        self.vocab_size = vocab_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transformer encoder. The input has dimensions (batch_size, sequence_length)
        and the output has dimensions (batch_size, sequence_length, embedding_dim).

        The encoder returns its output and the location of the padding, which will be
        used by the decoder.
        """
        # TODO: Implement the encoder (you should re-use the EncoderBlock class that we provided)

        #TODO: Padding embeddings of thr source may get used in the self-attention

        embeddings = self.embedding_layer(X)
        embeddings = self.position_encoding(embeddings)
        
        source_padding = (X != self.vocab_size) #shape is (batch_sz,source_seq_length)

        enc_out = embeddings
        mask = padding_mask(source_padding,source_padding)
        
        for b in self.blocks:
            enc_out = b(enc_out,mask=mask)

        return enc_out,source_padding

class DecoderBlock(Module):

    def __init__(self, embedding_dim, n_heads) -> None:
        super().__init__()

        self.attention1 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)
        self.attention2 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.dropout3 = Dropout(0.2)

    def forward(self, encoded_source: torch.Tensor, target: torch.Tensor,
                mask1: Optional[torch.Tensor]=None, mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of a decoder block. encoded_source has dimensions (batch_size, source_sequence_length, embedding_dim)
        and target has dimensions (batch_size, target_sequence_length, embedding_dim).

        The mask1 is passed to the first multi-headed self-attention layer, and mask2 is passed
        to the second multi-headed self-attention layer.

        Returns its output of shape (batch_size, target_sequence_length, embedding_dim) and
        the attention matrices for each of the heads of the second multi-headed self-attention layer
        (the one where the source and target are "mixed").
        """
        att_out, _ = self.attention1(target, target, target, mask1)
        residual = target + self.dropout1(att_out)

        X = self.norm1(residual)

        att_out, att_weights = self.attention2(X, encoded_source, encoded_source, mask2)

        residual = X + self.dropout2(att_out)
        X = self.norm2(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)
        residual = X + self.dropout3(temp)

        return self.norm3(residual), att_weights

class Decoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([DecoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])

        self.linear = Linear(embedding_dim, vocab_size + 1)
        self.softmax = Softmax(-1) #TODO: Not using softmax anywhere

        self.vocab_size = vocab_size

    def _lookahead_mask(self, seq_length: int) -> torch.Tensor:
        """
        Compute the mask to prevent the decoder from looking at future target values.
        The mask you return should be a tensor of shape (sequence_length, sequence_length)
        with only 1's and 0's, where a 0 represent an entry that will be masked in the
        multi-headed attention layer.

        Hint: The function torch.tril (https://pytorch.org/docs/stable/generated/torch.tril.html)
        may be useful.
        """
        # TODO: Implement the lookahead mask

        mask = torch.ones((seq_length,seq_length))
        mask = torch.tril(mask, diagonal=0) #TODO: don't consider elements above diagnol since we don't want to cheat/use future tokens

        return mask
    


    def forward(self, encoded_source: torch.Tensor, source_padding: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Transformer decoder. encoded_source has dimensions (batch_size, source_sequence_length, embedding),
        source_padding has dimensions (batch_size, source_seuqence_length) and target has dimensions
        (batch_size, target_sequence_length).

        Returns its output of shape (batch_size, target_sequence_length, target_vocab_size) and
        the attention weights from the first decoder block, of shape
        (batch_size, n_heads, source_sequence_length, target_sequence_length)

        Note that the output is not normalized (i.e. we don't use the softmax function).
        """
        # TODO: Implement the decoder (you should re-use the DecoderBlock class that we provided)

        batch_sz = target.shape[0]
        
        lookahead_mask1 = self._lookahead_mask(target.shape[1])[None] #(1,target_seq_len,target_seq_len)
        lookahead_mask1 = lookahead_mask1.repeat((batch_sz,1,1)) #(batch_sz,target_seq_len,target_seq_len)
       
        target_padding = (target != self.vocab_size)
        padding_mask1 = padding_mask(target_padding,target_padding)

        mask1 = padding_mask1*lookahead_mask1
        target = self.embedding_layer(target)
        target = self.position_encoding(target)
        
        mask2 = padding_mask(source_padding,target_padding)
        
        #TODO: Padding embeddings of the source may get used in the cross-attention - what should I use source-embeddings for?
        # Note: The source_padding passed in the unit tests is a tensor of 51s?
       

        dec_out = target
        for i in range(len(self.blocks)):
            dec_out,att_weights_i = self.blocks[i](encoded_source,dec_out,mask1=mask1,mask2=mask2)
            if i == 0:
                att_weights_0 = att_weights_i 

        dec_out = self.linear(dec_out)

        return dec_out,att_weights_0

class Transformer(Module):

    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int, n_encoder_blocks: int,
                 n_decoder_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.encoder = Encoder(source_vocab_size, embedding_dim, n_encoder_blocks, n_heads)
        self.decoder = Decoder(target_vocab_size, embedding_dim, n_decoder_blocks, n_heads)

    def forward(self, source, target):
        encoded_source, source_padding = self.encoder(source)
        return self.decoder(encoded_source, source_padding, target)

    def predict(self, source: List[int], beam_size=1, max_length=64) -> List[int]:
        """
        Given a sentence in the source language, you should output a sentence in the target
        language of length at most `max_length` that you generate using a beam search with
        the given `beam_size`.

        Note that the start of sentence token is 0 and the end of sentence token is 1.

        Return the final top beam (decided using average log-likelihood) and its average
        log-likelihood.

        Hint: The follow functions may be useful:
            - torch.topk (https://pytorch.org/docs/stable/generated/torch.topk.html)
            - torch.softmax (https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
        """
        self.eval() # Set the PyTorch Module to inference mode (this affects things like dropout)

        

        if not isinstance(source, torch.Tensor):
            source_input = torch.tensor(source).view(1, -1)
        else:
            source_input = source.view(1, -1)

        # TODO: Implement beam search.

        
        start_token = 2
        end_token=3

        iters = 1
        source_input = source_input.repeat(beam_size,1)
        target = torch.tensor([[start_token]]).repeat(beam_size,1)
        log_probs = torch.zeros(beam_size)
        completed_seq,completed_probs = [],[]

        encoded_source, source_padding = self.encoder(source_input)

        while iters < max_length and beam_size > 0:
            
            
            
            decoded_out,att_weights = self.decoder(encoded_source[:beam_size],source_padding[:beam_size],target)
            conditional_probs = torch.log(Softmax(dim=2)(decoded_out))[:,-1] #(beam_size,target_vocab_size)

            #import ipdb
            #ipdb.set_trace()

            log_probs, indices = torch.topk((log_probs[:,None] + conditional_probs).flatten(),k=beam_size) 


            indices = torch.tensor(np.unravel_index(indices, conditional_probs.shape))[1] #(beam_size)
            target = torch.cat([target,indices[:,None]],dim=1)

            iters += 1
            completed_seq += list(target[indices == end_token])
            completed_probs += list(log_probs[indices == end_token]*(1/iters))
            target = target[indices != end_token]
            log_probs = log_probs[indices != end_token]

            
            beam_size -=  torch.sum(indices == end_token)
            

    
        if beam_size > 0:
            completed_seq += list(target)
            completed_probs += list(log_probs*(1/max_length))
        

        min_index = torch.argmin(torch.tensor(completed_probs))
        min_avg_prob = completed_probs[min_index]
        seq = completed_seq[min_index]

        return seq.numpy(),min_avg_prob.detach()

def padding_mask(key_padding,query_padding):
    #query is target, key is source
    mask2 = key_padding[:,None] #shape is (batch_sz,1,source_seq_len) 
    target_seq_len = query_padding.shape[1]
    mask2 = mask2.repeat(1,target_seq_len,1) #shape is (batch_sz,target_seq_len,source_seq_len)
    mask2[query_padding == 0] = 0
    return mask2

        
def load_data() -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Dict[int, str], Dict[int, str]]:
    """ Load the dataset.

    :return: (1) train_sentences: list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) test_sentences : list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) source_vocab   : dictionary which maps from source word index to source word
             (3) target_vocab   : dictionary which maps from target word index to target word
    """
    with open('data/translation_data.bin', 'rb') as f:
        corpus, source_vocab, target_vocab = pickle.load(f)
        test_sentences = corpus[:1000]
        train_sentences = corpus[1000:]
        print("# source vocab: {}\n"
              "# target vocab: {}\n"
              "# train sentences: {}\n"
              "# test sentences: {}\n".format(len(source_vocab), len(target_vocab), len(train_sentences),
                                              len(test_sentences)))
        return train_sentences, test_sentences, source_vocab, target_vocab

def preprocess_data(sentences: Tuple[List[int], List[int]], source_vocab_size,
                    target_vocab_size, max_length):

    source_sentences = []
    target_sentences = []

    for source, target in sentences:
        source = [0] + source + ([source_vocab_size] * (max_length - len(source) - 1))
        target = [0] + target + ([target_vocab_size] * (max_length - len(target) - 1))
        source_sentences.append(source)
        target_sentences.append(target)

    return torch.tensor(source_sentences), torch.tensor(target_sentences)

def decode_sentence(encoded_sentence: List[int], vocab: Dict) -> str:
    if isinstance(encoded_sentence, torch.Tensor):
        encoded_sentence = [w.item() for w in encoded_sentence]
    words = [vocab[w] for w in encoded_sentence if w != 0 and w != 1 and w in vocab]
    return " ".join(words)

def visualize_attention(source_sentence: List[int],
                        output_sentence: List[int],
                        source_vocab: Dict[int, str],
                        target_vocab: Dict[int, str],
                        attention_matrix: np.ndarray):
    """
    :param source_sentence_str: the source sentence, as a list of ints
    :param output_sentence_str: the target sentence, as a list of ints
    :param attention_matrix: the attention matrix, of dimension [target_sentence_len x source_sentence_len]
    :param outfile: the file to output to
    """
    source_length = 0
    while source_length < len(source_sentence) and source_sentence[source_length] != 1:
        source_length += 1

    target_length = 0
    while target_length < len(output_sentence) and output_sentence[target_length] != 1:
        target_length += 1

    source_length += 1
    target_length += 1

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_matrix[:target_length, :source_length], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(source_length)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in source_vocab else source_vocab[x] for x in source_sentence[:source_length]]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(target_length)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in target_vocab else target_vocab[x] for x in output_sentence[:target_length]]))

    plt.show()
    plt.close()

def train(model: Transformer, train_source: torch.Tensor, train_target: torch.Tensor,
          test_source: torch.Tensor, test_target: torch.Tensor, target_vocab_size: int,
          epochs: int = 30, batch_size: int = 64, lr: float = 0.0001):

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss(ignore_index=target_vocab_size)

    epoch_train_loss = np.zeros(epochs)
    epoch_test_loss = np.zeros(epochs)

    for ep in range(epochs):

        train_loss = 0
        test_loss = 0

        permutation = torch.randperm(train_source.shape[0])
        train_source = train_source[permutation]
        train_target = train_target[permutation]

        batches = train_source.shape[0] // batch_size
        model.train()
        for ba in tqdm(range(batches), desc=f"Epoch {ep + 1}"):

            optimizer.zero_grad()

            batch_source = train_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = train_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        test_batches = test_source.shape[0] // batch_size
        model.eval()
        for ba in tqdm(range(test_batches), desc="Test", leave=False):

            batch_source = test_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = test_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            test_loss += batch_loss.item()

        epoch_train_loss[ep] = train_loss / batches
        epoch_test_loss[ep] = test_loss / test_batches
        print(f"Epoch {ep + 1}: Train loss = {epoch_train_loss[ep]:.4f}, Test loss = {epoch_test_loss[ep]:.4f}")

    return epoch_train_loss, epoch_test_loss

def strip_tokens(sentence):
    start_token,end_token = 2,3
    start_i = sentence.index(start_token)
    sentence = sentence[start_i+1:]
    end_i = sentence.index(end_token)
    sentence = sentence[:end_i]
    return sentence

def clipped_precision(predicted,target,N):
    pred_ngrams = defaultdict(int)
    target_ngrams = defaultdict(int)

    for i in range(len(predicted) - N + 1):
        pred_ngrams[tuple(predicted[i:i+N])] += 1

    for i in range(len(target) - N + 1):
        target_ngrams[tuple(target[i:i+N])] += 1

    score = 0
    
    for ngram in pred_ngrams:
        score += min(pred_ngrams[ngram],target_ngrams[ngram])
    
    score = score/(len(predicted) - N + 1)

    return score



def bleu_score(predicted: List[int], target: List[int], N: int = 4) -> float:
    """
    Implement a function to compute the BLEU-N score of the predicted
    sentence with a single reference (target) sentence.

    Please refer to the handout for details.

    Make sure you strip the SOS (2), EOS (3), and padding (anything after EOS)
    from the predicted and target sentences.

    If the length of the predicted sentence or the target is less than N,
    the BLEU score is 0.
    """
    # TODO: Implement bleu score

    
    predicted = strip_tokens(predicted)
    target = strip_tokens(target)

    if len(predicted) < N or len(target) < N:
        return 0
    
    score = 1
    for k in range(1,N+1):
        score = score*clipped_precision(predicted,target,k)**(1/N)
    brevity_penalty = min(1,np.exp(1 - (len(target)/len(predicted))))
    score = score*brevity_penalty

    return score
    


   

    


def seed_everything(seed=10707):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Loads data from English -> Spanish machine translation task
    import ipdb

    pe = PositionalEncodingLayer(24)
    X = torch.randn((5,50,21))
    pe.forward(X)

    import ipdb
    ipdb.set_trace()




    train_sentences, test_sentences, source_vocab, target_vocab = load_data()

    max_length = 64
    # Generates train/test data based on english and french vocabulary sizes and caps max length of sentence at 64
    train_source, train_target = preprocess_data(train_sentences, len(source_vocab), len(target_vocab), max_length)
    test_source, test_target = preprocess_data(test_sentences, len(source_vocab), len(target_vocab), max_length)
