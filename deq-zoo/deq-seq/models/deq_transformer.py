import torch
import torch.nn.functional as F
import torch.nn as nn

from torchdeq import get_deq, reset_deq
from torchdeq.dropout import VariationalDropout1d, VariationalDropout
from torchdeq.norm import apply_norm
from torchdeq.loss import fp_correction, jac_reg
from torchdeq.solver import solver_stat_from_final_step

from utils.adaptive_embedding import AdaptiveEmbedding
from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(self.inv_freq, pos_seq)    # C x L
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=0)   # Concat at feature dimension

        if bsz is not None:
            return pos_emb[None,:,:].expand(bsz, -1, -1)
        else:
            return pos_emb[None,:,:]


class Dropout(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training or not dropout:
            return x
        
        # Dimension (N, L, C)
        m = torch.zeros_like(x[:,:1]).bernoulli_(1 - dropout)
        mask = m.requires_grad_(False) / (1 - dropout)
        mask = mask.expand_as(x).to(x)
        return mask * x


class LayerNorm(nn.Module):
    def __init__(self, shape, dim=1):
        super().__init__()
        
        if type(shape) not in (list, tuple):
            shape = [shape]

        self.shape = shape
        self.dim = dim

    def forward(self, x):
        if self.dim != -1:
            x = x.transpose(self.dim, -1)
        x = F.layer_norm(x, self.shape)
        if self.dim != -1:
            x = x.transpose(self.dim, -1)

        return x


class FFN(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, kernel=1):
        super(FFN, self).__init__()
        
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.ff1_net = nn.Linear(d_model, d_inner)
        self.drop1 = VariationalDropout1d(dropout=dropout)

        self.ff2_net = nn.Linear(d_inner, d_model)
        self.drop2 = VariationalDropout1d(dropout=dropout)

        self.pre_lnorm = pre_lnorm
        self.ln = LayerNorm(self.d_model, dim=-1)
    
    def forward(self, inp, attn_out=None):
        # (B, D, L) -> (B, L, D)
        inp = inp.transpose(1, 2)
        ffn_inp = inp

        if self.pre_lnorm:
            ffn_inp = self.ln(ffn_inp)

        relu_out1 = self.drop1(F.relu(self.ff1_net(ffn_inp)))
        out2 = self.drop2(self.ff2_net(relu_out1))

        output = out2 + inp
        if not self.pre_lnorm:
            output = self.ln(output)
        
        # (B, L, D) -> (B, D, L)
        output = output.transpose(1, 2)

        return output


class Attention(nn.Module):
    # This is similar to the RelPartialLearnableMultiHeadAttn class in Transformer-XL
    def __init__(self, d_model, n_head, d_head, dropout, dropatt, 
                 kernel=1, pre_lnorm=False, local_size=None):
        super(Attention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.scale = 1 / (d_head ** 0.5)

        self.qkv_net = nn.Conv1d(d_model, 3 * n_head * d_head, kernel, bias=False)
        self.r_net = nn.Conv1d(d_model, n_head * d_head, kernel, bias=False)
        
        self.r_w_bias = nn.Parameter(torch.rand(n_head, d_head).uniform_(-0.05, 0.05))
        self.r_r_bias = nn.Parameter(torch.rand(n_head, d_head).uniform_(-0.05, 0.05))
        
        self.o_net = nn.Conv1d(n_head * d_head, d_model, kernel)
        
        self.dropatt = VariationalDropout(dropout=dropatt)
        self.drop = VariationalDropout1d(dropout=dropout, token_first=False)

        self.pre_lnorm = pre_lnorm
        self.ln = LayerNorm(self.d_model, dim=1)

        self.local_size = local_size
        
    def _rel_shift(self, x):
        # x has dimension (bsz x n_head x qlen x klen)
        bsz, n_head, qlen, klen = x.shape
        x_padded = F.pad(x, (1,0))
        x_padded = x_padded.view(bsz, n_head, klen+1, qlen)
        return x_padded[:,:,1:].view_as(x)

    def forward(self, z, pos_emb, u, mems=None):
        # Note: In this context, qlen means the length of the sequence; and mlen describes
        #       the length of the padding. Their sum is klen. 
        
        bsz, d_model, qlen = z.shape

        r_w_bias, r_r_bias = self.r_w_bias, self.r_r_bias
        n_head, d_head = self.n_head, self.d_head
        rlen = pos_emb.shape[2]
        
        if mems is None: 
            mems = torch.tensor([]).view(0, 0, 0)
        mlen = mems.shape[2]
        cat = torch.cat([mems, z], dim=-1)

        if self.pre_lnorm:
            cat = self.ln(cat)
        
        w_heads = self.qkv_net(cat)      # (N x 3*d_model x seq_len)
        r_head_k = self.r_net(pos_emb)

        # Input injection
        w_heads += u
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=1)
        w_head_q = w_head_q[:,:,-qlen:]

        klen = w_head_k.shape[2]

        w_head_q = w_head_q.view(bsz, n_head, d_head, qlen)           # bsz x n_head x d_head x qlen
        w_head_k = w_head_k.view(bsz, n_head, d_head, klen)           # bsz x n_head x d_head x klen
        w_head_v = w_head_v.view(bsz, n_head, d_head, klen)           # bsz x n_head x d_head x klen

        r_head_k = r_head_k.view(n_head, d_head, rlen)                # n_head x d_head x rlen

        # Compute attention score
        rw_head_q = w_head_q + r_w_bias[:,:,None]                   # bsz x n_head x d_head x qlen
        AC = torch.einsum('bndi,bndj->bnij', rw_head_q, w_head_k)
        rr_head_q = w_head_q + r_r_bias[:,:,None]
        BD = torch.einsum('bndi,ndj->bnij', rr_head_q, r_head_k)
        BD = self._rel_shift(BD)    # for relative positional embedding

        attn_score = AC + BD        # bsz x n_head x qlen x klen
        attn_score.mul_(self.scale)
            
        # Compute attention probability
        # We apply a local mask, with local horizon size of mlen
        local_size = self.local_size or 1000
        attn_mask = (torch.triu(torch.ones(qlen, klen), diagonal=1+mlen) > 0)[None]
        attn_mask += (torch.tril(torch.ones(qlen, klen), diagonal=mlen-local_size) > 0)[None]
        if attn_mask is not None and attn_mask.any().item():
            attn_score = attn_score.float().masked_fill(
                    attn_mask[None], -float('inf')).type_as(attn_score)
                
        attn_prob = F.softmax(attn_score, dim=-1)          # bsz x n_head x qlen x klen
        attn_prob = self.dropatt(attn_prob)
            
        # Compute attention vector
        attn_vec = torch.einsum('bnij,bndj->bndi', (attn_prob, w_head_v))
        
        # [bsz x d x qlen]
        attn_vec = attn_vec.reshape(bsz, n_head*d_head, attn_vec.size(-1))

        # Linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        
        # Residual connection + layer normolization (if applicable)
        out = attn_out + z
        if not self.pre_lnorm:
            out = self.ln(out)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(TransformerDecoder, self).__init__()

        pre_lnorm = kwargs.get('pre_lnorm')
        local_size = kwargs.get('local_size', None)
        dropatt = kwargs.get('dropatt', 0.0)
        self.dec_attn = Attention(
                d_model, n_head, d_head, dropout=dropout, dropatt=dropatt, 
                pre_lnorm=pre_lnorm,  local_size=local_size
                )
        self.pos_ff = FFN(d_model, d_inner, dropout, pre_lnorm=pre_lnorm)
    
    def forward(self, z_now, z_hist, u, pos_emb, **kwargs):
        output = self.dec_attn(z_now, pos_emb, u, mems=z_hist)
        output = self.pos_ff(output)

        return output


class DEQTransformerLM(nn.Module):
    def __init__(self, args, n_token, n_layer, eval_n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weights=True, d_embed=None, div_val=1,
                 tie_projs=[False], pre_lnorm=False, wnorm=False, tgt_len=None,
                 mem_len=None, local_size=0, cutoffs=[], logging=None):
        super().__init__()
        self.args = args
        self.logging = logging or print

        self.n_token = n_token
        
        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)
        self.iodrop = Dropout(dropout=dropout)
        self.pos_drop = VariationalDropout1d(dropout=dropout)
        
        assert mem_len > 0
        self.mem_len = mem_len
        self.local_size = local_size
        self.max_klen = tgt_len + mem_len

        self.n_layer = n_layer
        self.eval_n_layer = eval_n_layer

        self.inject_conv = nn.Conv1d(d_model, 3*d_model, kernel_size=1)
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.func = TransformerDecoder(
                n_head, d_model, d_head, d_inner, dropout=dropout, dropatt=dropatt,
                pre_lnorm=pre_lnorm, local_size=local_size
                )
        
        apply_norm(self.func, args=args)
        self.deq = get_deq(args)

        # use adaptive softmax (including standard softmax)
        # (Note: To use sample softmax, refer to the Transformer-XL implementation)
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
        return mems       # For z_hist and u_hist
    
    @torch.no_grad()
    def _update_mems(self, z_now, z_hist, u_cat, qlen, mlen):
        end_idx = mlen + qlen
        beg_idx = max(0, end_idx - self.mem_len)    # Account for when mlen = 0
        z_cat = torch.cat([z_hist, z_now], dim=2)

        new_z_hist = z_cat[:,:,beg_idx:end_idx]     # (B, D, L) 
        new_u_hist = u_cat[:,:,beg_idx:end_idx]

        return [new_z_hist, new_u_hist]

    def forward(self, data, target, mems, **kwargs):
        # nn.DataParallel does not allow shape[0] tensors to be broadcasted.
        # So, have to initialize shape[0] mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        '''
        assume both data and target are of the shape (B, L)
        mems (B, D, L)
        '''
        if not mems: 
            mems = self.init_mems()
        
        bsz, qlen = data.shape
        tgt_len = target.shape[-1]

        mlen = 0 if mems[0].nelement() == 0 else mems[0].shape[2]
        klen = mlen + qlen
    
        # Reset normalization and dropout masks in DEQ
        reset_deq(self)
 
        compute_jac_loss = kwargs.get('compute_jac_loss', True)
        sradius_mode = kwargs.get('sradius_mode', False)
        writer = kwargs.get('writer', None)
        
        word_emb = self.word_emb(data)
        word_emb = self.iodrop(word_emb)
        u_now = self.inject_conv(word_emb.transpose(1,2))      # bsz x 3*d_model x qlen

        z_hist, u_hist = mems
        d_model = self.d_model
        if z_hist is not None and z_hist.nelement() > 0:
            assert z_hist.shape[2] == u_hist.shape[2], "Padding fixed points and padding embedding dimensions don't agree"
        else:
            z_hist, u_hist = torch.zeros(bsz, d_model, 0), torch.zeros(bsz, 3*d_model, 0)
        mlen = z_hist.shape[2]
        klen = mlen + qlen    # qlen is seq_len, mlen is pad_len

        pos_seq = torch.arange(klen-1, -1, -1.0)
        pos_emb = self.pos_drop(self.pos_emb(pos_seq))                  # bsz x d_model x (qlen + mlen) for positional embedding
        u_cat = torch.cat([u_hist, u_now], dim=2)

        z_now = torch.zeros(bsz, d_model, qlen, device=data.device)     # (B, D, L) for initial estimate of output
        jac_loss = torch.zeros(bsz, 1, device=data.device)
        sradius = torch.zeros(bsz, 1, device=data.device)

        def deq_func(z_now):
            return self.func(z_now, z_hist, u_cat, pos_emb)

        z_out, info = self.deq(
                deq_func, z_now, 
                sradius_mode=sradius_mode, 
                backward_writer=writer
                )
        z_now = z_out[-1]
        
        if self.training and compute_jac_loss:
            jac_loss = jac_reg(deq_func(z_now), z_now)

        def decode(z):
            z_pred = self.iodrop(z)   # (B, D, L) 
            z_pred = z_pred[:,:,-tgt_len:]        
            return z_pred

        z_pred = [decode(z) for z in z_out]
        loss = fp_correction(self.crit, (z_pred, target), gamma=self.args.gamma)

        new_mems = self._update_mems(z_now, z_hist, u_cat, mlen, qlen)
        return loss, jac_loss, sradius, new_mems, info
