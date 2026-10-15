from transformers import PretrainedConfig

class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import math
from typing import Optional,Tuple
from torch.nn import functional as F
from .activation_functions import ACT2FN
#################            rmsnorm            #################
#先理清思路，rmsnorm是一层，就需要继承nn.Module类
class RMSNorm(nn.Module):

#既然是类的话就要编写__init__初始化
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim#维度
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
        
#然后编写_norm计算公式
    def _norm(self,x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*x

#但nn.Module类需要强制实现forward方法、实现前向传播
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)
    

#################            rope/yarn            #################
def precompute_freqs_cis(dim:int,end:int(32*1024),rope_base,rope_scaling:Optional[dict]=None):#end是整个可能处理的序列长度  
    #初始化rope频率
    freqs,attn_factor=(1.0/(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim)),1.0)

    if rope_scaling is not None:#意思是需要用到yarn时
        orig_max,factor,beta_fast,beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
        )
        #推断的长度大于训练长度，使用缩放
        if end>orig_max:
            #波长b到i的映射
            inv_dim=lambda b:(dim * math.log(orig_max / (b*2*math.pi))) /(2*math.log(rope_base))

            #划分高低维度
            #low：不需要缩放的高频部分
            #high：需要缩放的低频部分
            low,high=(max(math.floor(inv_dim(beta_fast)),0),min(math.ceil(inv_dim(beta_slow)),dim//2-1))

            #计算缩放因子
            # low之前，ramp为0，在high之后，ramp为1；在low和high之间，ramp为0到1的线性变化
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp 在 0–1 之间时：平滑过渡。
            freqs=freqs*(1-ramp+ramp*factor)
        #根据end，生成位置索引t
        t=torch.arange(end,device=freqs.device).float()

        # 计算外积，将 t 和频率部分相乘，得到每个位置的旋转角度
        freqs = torch.outer(t, freqs).float()

        freqs_cos = (
        torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
        )
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

        return freqs_cos, freqs_sin


#################            RoPE            #################
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # [a, b] -> [-b, a]
    def rotate_half(x):
        # x.shape[-1] 取最后一个维度的长度
        # x[..., x.shape[-1] // 2 :] 取出 x 的后半部分
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    # x_rotated = x * cos + rotate_half(x) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed
  

#################            GQA            #################

#因为GQA需要重复用到kv，所以先写一个repeat_kv函数，重复kv
def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    #x是tensor，n_rep是重复的次数
    bs,slen,num_key_value_hands,head_dim=x.shape
    if n_rep==1:
        return x

    #在第四个维度插入一个新的维度，然后在这个维度上重复n_rep次
    return (x[:,:,:,None,:]                                 #四个维度插入一个新的维度(bs, slen, num_key_value_heads, 1, head_dim)
    .expand(bs,slen,num_key_value_hands,n_rep,head_dim)     #扩展复制维度，如果n_rep=2，那么就会在第四个维度上复制两次(bs, slen, num_key_value_heads, 2 , head_dim)
    .reshape(bs,slen,num_key_value_hands*n_rep,head_dim)    #合并头的维度(bs, slen, 2*num_key_value_heads , head_dim),num_key_value_heads 个 Key/Value 头被扩展成了2倍个，每个头重复 2 次。
    )

 
class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        # 取key/value头的数量，如果args.num_key_value_heads为None，则使用args.num_attention_heads作为默认值，否则使用参数里的args.num_key_value_heads
        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is None else args.num_attention_heads

        #q一定是key和value的整数倍，不然输出报错
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        #设置配置
        #初始化当前这个实例所使用的头的数量
        self.n_local_heads = args.num_attention_heads
        #这个实例使用的key和value的数量等于参数里定义的key和value的数量
        self.num_key_value_heads = args.num_key_value_heads
        #重复的key和value的次数等于参数里定义的key和value的数量除以当前实例使用的头的数量
        self.n_rep=self.num_local_heads // self.num_key_value_heads
        #每个头的维度等于隐藏层的维度除以当前实例使用的头的数量
        self.head_dim = args.hidden_size // args.num_attention_heads


        #对线性层的qkv进行定义，乘以w（投影层）
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads*self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads*self.head_dim, bias=False)
        #输出也有一个线性层，需要用output把前面的数量都拼回来
        self.o_proj = nn.Linear(args.num_attention_heads*self.head_dim, args.hidden_size, bias=False)

        #其他要用到的参数
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        #是否使用flash attention，flash attention是一种优化的注意力机制，可以减少内存占用和计算量，提高训练速度。
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention

    def forward(
        self,
        x: torch.Tensor,
        position_embedding: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

    # 投影，计算q,k,v
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    # 把输入拆分成多个头，用view
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
    # q和k：使用roPE
        cos, sin = position_embedding
        q, k = apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])
    # 对于k和v，使用repeat（注意kv cache）
        # q、k 已分头，且已对 q、k 应用 RoPE
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
    # 进行attention计算，q@k^T/sqrt(d)
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1)
                .expand(bsz, self.n_local_heads, seq_len, -1)
                .bool()
            )
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # [bsz, n_local_heads, seq_len, head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


#################            FFN            #################
# FFN是一层
class FeedForward(nn.Module):
    #初始化
    def __init__(self,args:MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:#如果没有指定中间层的大小，就按照hidden_size*8/3来计算
            args.intermediate_size = int(args.hidden_size*8/3)
            args.intermediate_size = 64*((intermediate_size+64-1)//64)
        #实现投影,升维,降维,门控,dropout,激活函数
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

        #前向传播
    def forward(self, x):
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x) * self.up_proj(x)))
    )

#Block拼接
class MokioMindBlock(nn.module):
    def __init__(self,layer_id:int,config:MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
    self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value
        

#组装Model
class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [MokioMindBlock(i, config) for i in range(self.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #Rope预计算
        freqs_cos, freqs_sin = precomput_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)

        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len],
        )

        presents = []
        for layer_idx, (layer, past_key_values) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )

            presents.append(present)

        hidden_states=self.norm(hidden_states)

        return hidden_states,presents
















