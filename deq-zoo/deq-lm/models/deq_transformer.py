import torch
import torch.nn.functional as F
import torch.nn as nn

from torchdeq import get_deq, reset_deq
from torchdeq.dropout import VariationalDropout1d, VariationalDropout
from torchdeq.loss import fp_correction
from torchdeq.utils.mem import mem_gc
from torchdeq.solver import solver_stat_from_final_step

from utils.adaptive_embedding import AdaptiveEmbedding
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax


class PositionalEmbedding(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_embed, 2.0) / d_embed))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)                      # (L, D)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)    # Concat at feature dimension
        return pos_emb[None,:,:]


class Dropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x, drop_dim=1):
        if not self.training or not self.dropout:
            return x
        
        # Dimension (N, L, D)
        shape_tensor = x.narrow(drop_dim, 0, 1)
        m = torch.zeros_like(shape_tensor).bernoulli_(1 - self.dropout)

        mask = m.requires_grad_(False) / (1 - self.dropout)
        mask = mask.expand_as(x).to(x)
        return mask * x


class FFN(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.ff1_net = nn.Linear(d_model, d_inner)
        self.drop1 = VariationalDropout1d(dropout=dropout)

        self.ff2_net = nn.Linear(d_inner, d_model)
        self.drop2 = VariationalDropout1d(dropout=dropout)

        self.ln = nn.LayerNorm(self.d_model)
    
    def forward(self, x):
        inp = self.ln(x)

        relu_out1 = self.drop1(F.relu(self.ff1_net(inp)))
        out2 = self.drop2(self.ff2_net(relu_out1))

        output = x + out2

        return output


class Attention(nn.Module):
    def __init__(self, d_model, n_head, d_head, dropout, dropatt):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.scale = 1 / (d_head ** 0.5)

        self.qkv_net = nn.Linear(d_model, 3*n_head*d_head, bias=False)
        self.r_net = nn.Linear(d_model, n_head*d_head, bias=False)
        
        self.r_w_bias = nn.Parameter(torch.rand(n_head, d_head).uniform_(-0.05, 0.05))
        self.r_r_bias = nn.Parameter(torch.rand(n_head, d_head).uniform_(-0.05, 0.05))
        
        self.o_net = nn.Linear(n_head*d_head, d_model)
        
        self.dropatt = VariationalDropout(dropout=dropatt)
        self.drop = VariationalDropout1d(dropout=dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def _rel_shift(self, x):
        bsz, n_head, q_len, k_len = x.shape
        x_padded = F.pad(x, (1,0))
        x_padded = x_padded.view(bsz, n_head, k_len+1, q_len)
        return x_padded[:,:,1:].view_as(x)

    def forward(self, z, z_hist, pos_emb, u, attn_mask=None):
        bsz, q_len, d_model = z.shape                                   # (B, L_q, D)
        n_head, d_head = self.n_head, self.d_head
        
        # Cat memory
        cat = torch.cat([z_hist, z], dim=1)                             # L_k = L_q + L_m
        m_len = z_hist.shape[1]
        k_len = cat.shape[1]

        # PreLN + Linear
        cat = self.ln1(cat)
        w_heads = self.qkv_net(cat)                                     # (B, L_k, 3*H*D_h)
        r_head_k = self.r_net(pos_emb)                                  # (L_k, H*D_h)

        # Input injection
        w_heads += u
        
        w_heads = w_heads.view(bsz, k_len, n_head, 3*d_head)
        r_head_k = r_head_k.view(k_len, n_head, d_head)
        
        # Keep q_len only
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_head_q[:, -q_len:]

        # Compute attention score
        rw_head_q = w_head_q + self.r_w_bias                            # (B, L_q, H, D)
        AC = torch.einsum('bind,bjnd->bnij', rw_head_q, w_head_k)       # (B, H, L_q, L_k)
        
        # Rel Positional Embedding
        rr_head_q = w_head_q + self.r_r_bias                            # (B, L_q, H, D)
        BD = torch.einsum('bind,jnd->bnij', rr_head_q, r_head_k)        # (B, H, L_q, L_k)
        
        # for relative positional embedding
        BD = self._rel_shift(BD)

        attn_score = AC + BD                                            # (B, H, L_q, L_k)
        attn_score.mul_(self.scale)
            
        # Apply the local mask, with local horizon size of m_len
        if attn_mask is not None and attn_mask.any().item():
            attn_score = attn_score.float().masked_fill(
                    attn_mask, -float('inf')).type_as(attn_score)

        attn_prob = F.softmax(attn_score, dim=-1)                       # (B, H, L_q, L_k)
        attn_prob = self.dropatt(attn_prob)
            
        # Compute attention vector
        attn_vec = torch.einsum('bnij,bjnd->bind', attn_prob, w_head_v)
        attn_vec = attn_vec.reshape(bsz, q_len, n_head*d_head)
        
        # PostAttn LayerNorm
        attn_out = self.ln2(attn_vec)
        attn_out = self.o_net(attn_out)
        attn_out = self.drop(attn_out)
        
        # Residual connection
        out = attn_out + z

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, args, n_head, d_model, d_head, d_inner, dropout, dropatt, n_layer=1, **kwargs):
        super(TransformerDecoder, self).__init__()

        self.n_layer = n_layer

        self.mem = args.mem

        self.dec_attn = nn.ModuleList([
            Attention(d_model, n_head, d_head, dropout=dropout, dropatt=dropatt)
            for _ in range(n_layer)
            ])
        self.pos_ff = nn.ModuleList([
            FFN(d_model, d_inner, dropout)
            for _ in range(n_layer)
            ])

    def forward(self, z_now, z_hist, u, pos_emb, attn_mask, **kwargs):
        for i in range(self.n_layer):            
            if self.mem:
                z_now = mem_gc(self.dec_attn[i], (z_now, z_hist, pos_emb, u, attn_mask))
                z_now = mem_gc(self.pos_ff[i], (z_now,))
            else:
                z_now = self.dec_attn[i](z_now, z_hist, pos_emb, u, attn_mask)
                z_now = self.pos_ff[i](z_now)

        return z_now


class DEQTransformerLM(nn.Module):
    def __init__(self, args, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, d_embed=None, tgt_len=None, mem_len=None, local_size=0,
                 tie_weights=True, div_val=1, tie_projs=[False], cutoffs=[], logging=None):
        super().__init__()
        self.args = args
        self.logging = logging or print

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
       
        assert mem_len > 0
        self.mem_len = mem_len
        self.local_size = local_size
        self.max_k_len = tgt_len + mem_len

        # Injection
        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)
        self.pos_emb = PositionalEmbedding(d_model)
        self.injection = nn.Linear(d_model, 3*n_head*d_head)
        
        # DEQ
        self.func = TransformerDecoder(
                args, n_head, d_model, d_head, d_inner, dropout=dropout, dropatt=dropatt, n_layer=n_layer
                )
        self.deq = get_deq(args)

        self.pos_drop = VariationalDropout1d(dropout=dropout)
        self.iodrop = Dropout(dropout=dropout)
        
        # Decoder
        # use adaptive softmax (including standard softmax)
        self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs, div_val=div_val)
        
        if tie_weights:
            for i in range(len(self.crit.out_layers)):
                self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight
        
        if tie_projs:
            for i, tie_proj in enumerate(tie_projs):
                if tie_proj and div_val == 1 and d_model != d_embed:
                    self.crit.out_projs[i].weight.data = self.word_emb.emb_projs[0].weight.data
                elif tie_proj and div_val != 1:
                    self.crit.out_projs[i].weight.data = self.word_emb.emb_projs[i].weight.data

    @torch.no_grad()
    def init_mems(self):
        mems = [torch.empty(0), torch.empty(0)]
        return mems                                                     # For z_hist and u_hist
    
    @torch.no_grad()
    def _update_mems(self, z_now, z_hist, u_cat, q_len, m_len):
        end_idx = m_len + q_len
        beg_idx = max(0, end_idx - self.mem_len)                        # Account for when m_len = 0
        z_cat = torch.cat([z_hist, z_now], dim=1)

        new_z_hist = z_cat[:,beg_idx:end_idx]                           # (B, L_k, D) 
        new_u_hist = u_cat[:,beg_idx:end_idx]

        return [new_z_hist, new_u_hist]

    def forward(self, data, target, mems, **kwargs):
        '''
        assume both data and target (B, L), mems (B, L, D)
        '''
        if not mems: 
            mems = self.init_mems()
        
        bsz, q_len = data.shape
        tgt_len = target.shape[-1]
        m_len = 0 if mems[0].nelement() == 0 else mems[0].shape[1]
        k_len = m_len + q_len

        d_model, n_head, d_head = self.d_model, self.n_head, self.d_head

        # Reset normalization and dropout masks in DEQ
        reset_deq(self)

        word_emb = self.word_emb(data)
        word_emb = self.iodrop(word_emb)
        u_now = self.injection(word_emb)                                # (B, L_q, H*D_h)

        z_hist, u_hist = mems
        if z_hist is not None and z_hist.nelement() > 0:
            assert z_hist.shape[1] == u_hist.shape[1], "Padding fixed points and padding embedding dimensions don't agree"
        else:
            z_hist, u_hist = torch.zeros(bsz, 0, d_model), torch.zeros(bsz, 0, 3*n_head*d_head)

        m_len = z_hist.shape[1]
        k_len = q_len + m_len                                           # L_k = L_q + L_m

        pos_seq = torch.arange(k_len-1, -1, -1.0)
        pos_emb = self.pos_drop(self.pos_emb(pos_seq))                  # (B, L_k, D)
        u_cat = torch.cat([u_hist, u_now], dim=1)                       # (B, L_k, 3*H*D_h)
        
        # Generate Attetion mask
        local_size = self.local_size or 1000
        attn_mask = (torch.triu(torch.ones(q_len, k_len), diagonal=1+m_len) > 0)
        attn_mask += (torch.tril(torch.ones(q_len, k_len), diagonal=m_len-local_size) > 0)

        z_now = torch.zeros(bsz, q_len, d_model, device=data.device)    # (B, D, L)
        def deq_func(z_now):
            return self.func(z_now, z_hist, u_cat, pos_emb, attn_mask)

        z_out, info = self.deq(deq_func, z_now)
        z_now = z_out[-1]
        
        with torch.no_grad():
            z_next = deq_func(z_now)
            info = solver_stat_from_final_step(z_now, z_next, self.args.grad[-1])

        def decode(z):
            z_pred = self.iodrop(z, drop_dim=-1)                        # (B, L_q, D) 
            z_pred = z_pred[:,-tgt_len:]                                # (B, L_t, D)
            return z_pred
        
        z_pred = [decode(z) for z in z_out]
        loss = fp_correction(self.crit, (z_pred, target), gamma=self.args.gamma)

        new_mems = self._update_mems(z_now, z_hist, u_cat, m_len, q_len)
        return loss, new_mems, info
