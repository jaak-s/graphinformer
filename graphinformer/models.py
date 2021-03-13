import torch
import math

class GIConfig(object):
    def __init__(self,
            num_heads,
            key_size,
            key_r_size,
            value_size,
            hidden_size,
            intermediate_size,
            route_size,
            node_feature_size,
            num_node_types,
            attention_dropout,
            hidden_dropout,
            embedding_dropout,
            num_layers,
            initializer_range,
            init_norm = 0.01,
            init_norm_emb    = None,
            head_radius      = None,
            use_route_values = False,
            regression_loss  = None,
            regression_size  = None,
            graph_num_labels = None,
            graph_num_values = None,
            channel_dropout  = 0.0,
            cdo_schedule     = 0,
            final_dropout    = 0.0,
            pooler_pre_layer = True,
            pooler_post_layer= False,
            pooler_type      = "mean",
            weight_decay     = 0.0,
            out_num_heads    = None,
            out_value_size   = None,
            out_hidden_size  = None,
            embed_routes     = False,
            route_feature_size = None,
            residual         = True,
            learning_rate    = None,
            num_epochs       = None,
            batch_size       = None,
            input_size       = None,
            injective        = False, #To make the GraphInformer injective (sigmoid instead of softmax for attention scores)
        ):
        self.num_heads         = num_heads
        self.key_size          = key_size
        self.key_r_size        = key_r_size
        self.value_size        = value_size
        self.hidden_size       = hidden_size
        self.intermediate_size = intermediate_size
        self.route_size        = route_size
        self.node_feature_size = node_feature_size

        self.num_node_types    = num_node_types

        self.attention_dropout = attention_dropout
        self.hidden_dropout    = hidden_dropout
        self.embedding_dropout = embedding_dropout
        self.num_layers        = num_layers

        self.initializer_range = initializer_range
        self.init_norm         = init_norm
        self.use_route_values  = use_route_values

        self.head_radius = head_radius
        if head_radius is not None:
            assert len(head_radius) == num_heads

        self.regression_loss = regression_loss
        self.regression_size = regression_size

        self.graph_num_labels = graph_num_labels
        self.graph_num_values = graph_num_values

        self.channel_dropout  = channel_dropout
        self.cdo_schedule     = cdo_schedule
        self.final_dropout    = final_dropout
        self.pooler_pre_layer = pooler_pre_layer
        self.pooler_post_layer= pooler_post_layer
        self.pooler_type      = pooler_type

        self.weight_decay     = weight_decay

        self.out_num_heads    = out_num_heads
        self.out_value_size   = out_value_size
        self.out_hidden_size  = out_hidden_size

        self.injective        = injective

        self.embed_routes     = embed_routes
        self.route_feature_size = route_feature_size
        if not self.embed_routes:
            self.route_feature_size = self.route_size
        else:
            assert self.route_size is not None

        if init_norm_emb is None:
            self.init_norm_emb = init_norm
        else:
            self.init_norm_emb = init_norm_emb

        self.residual         = residual
        self.learning_rate    = learning_rate
        self.num_epochs       = num_epochs
        self.batch_size       = batch_size
        self.input_size       = input_size

############## toy code (no batching) #############
class GraphAttentionToy(torch.nn.Module):
    def __init__(self, din, dk, dv, dr):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.dr = dr
        self.W  = torch.nn.Linear(din, dk + dk + dv + dr, bias=False)

    def forward(self, H, D):
        """
        Args:
          H   hiddens (n x d)
          D   route information (n x n x dr)
        """
        Q, K, V, R = torch.split(self.W(H), [self.dk, self.dk, self.dv, self.dr], dim=1)
        ## compute route attention. R is [n x dr] and D is [n x n x dr]:
        RD      = torch.bmm(R.unsqueeze(1), D.permute(0, 2, 1)).squeeze(1)
        scale   = 1.0 / math.sqrt(self.dk + self.dr)
        weights = scale * (Q @ K.transpose(1, 0) + RD)

        return weights.softmax(dim=1) @ V

############## toy code (no batching) #############
class MHGraphAttention1(torch.nn.Module):
    def __init__(self, din, dk, dv, dr, h, dmodel, dropout):
        """
        Args:
          h   number of heads
        """
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.dr = dr
        self.h  = h
        self.W  = torch.nn.Linear(din, h * (dk + dk + dv + dr), bias=False)
        self.Wout = torch.nn.Linear(dv * h, dmodel, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, H, D, attention_mask):
        ## TODO: add batch infront of H and D, plus a list of lengths
        """
        Args:
          H   hiddens [n x din]
          D   route information [n x n x dr]
          attention mask   0 if allowed -Inf for not allowed [n x n]
        """
        H2 = self.W(H).view(-1, self.h, self.dk + self.dk + self.dv + self.dr).transpose(0, 1)
        ## H2 is [h x n x dsum]
        Q, K, V, R = torch.split(H2, [self.dk, self.dk, self.dv, self.dr], dim=-1)

        ## compute route attention. R is [num_heads, n, dr] and D is [n, n, dr]:
        ## R => [num_heads, n,  1, dr]
        ## D =>            [n, dr, n]
        RD      = torch.matmul(R.unsqueeze(2), D.transpose(-1, -2)).squeeze(2)
        scale   = 1.0 / math.sqrt(self.dk + self.dr)
        attention_scores = scale * (Q @ K.transpose(-1, -2) + RD) + attention_mask

        ## context_layer is [h x n x dv] => [n x h*dv]
        context_layer = (attention_scores.softmax(dim=-1) @ V).transpose(0, 1).contiguous().view(-1, self.h * self.dv)
        context_layer = self.dropout(context_layer)

        return self.Wout(context_layer)


class GIEmbeddings(torch.nn.Module):
    """
    Args:
        config         configuration object

    The module can be called with following inputs:
        node_ids       int tensor [batch, seq_pos] each batch has several numbers
        node_features  float tensor [batch, seq_pos, node_feature]
    """
    def __init__(self, config):
        super().__init__()
        self.node_embeddings = torch.nn.Embedding(config.num_node_types, config.hidden_size, padding_idx=0)
        self.layer_norm      = LayerNorm(config.hidden_size, eps=1e-12, init_norm=config.init_norm_emb)
        self.dropout         = torch.nn.Dropout(config.embedding_dropout)
        ## TODO: maybe linear is better here (or just linear + tanh/relu)
        self.feature_net     = torch.nn.Sequential(
            torch.nn.Linear(config.node_feature_size, config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, node_ids, node_features):
        ## node_ids is [batch, n]
        ## node_ids is [batch, n, node_feature_size]
        node_embeddings = self.node_embeddings(node_ids)
        node_hidden     = self.feature_net(node_features)
        #embeddings      = self.layer_norm(node_embeddings + node_hidden)
        #embeddings      = self.dropout(embeddings)
        embeddings      = self.layer_norm(self.dropout(node_embeddings + node_hidden))
        return embeddings
        
class GIRouteEmbeddings(torch.nn.Module):
    """
    Args:
        config         configuration object

    The module can be called with following inputs:
        route_data     float tensor [batch, seq_pos, seq_pos, route_feature]
    """
    def __init__(self, config):
        super().__init__()
        self.layer_norm      = LayerNorm(config.route_size, eps=1e-12, init_norm=config.init_norm_emb)
        self.dropout         = torch.nn.Dropout(config.embedding_dropout)
        ## TODO: maybe linear is better here (or just linear + tanh/relu)
        self.feature_net     = torch.nn.Sequential(
            torch.nn.Linear(config.route_feature_size, config.route_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.route_size, config.route_size),
        )

    def forward(self, route_data):
        ## route_data is [batch, seq_pos, seq_pos, route_feature]
        route_hidden = self.feature_net(route_data)
        embeddings   = self.layer_norm(self.dropout(route_hidden))
        return embeddings

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12, init_norm=None):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.init_norm = init_norm

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GraphSelfAttentionOld(torch.nn.Module):
    def __init__(self, config):
        """
        Args:
          h   number of heads
        """
        super().__init__()
        self.num_heads   = config.num_heads
        self.key_size    = config.key_size
        self.value_size  = config.value_size
        self.hidden_size = config.hidden_size
        self.route_size  = config.route_size

        total_size   = self.key_size + self.key_size + self.value_size + self.route_size
        self.W       = torch.nn.Linear(self.hidden_size, self.num_heads * total_size, bias=False)
        #self.Wout    = torch.nn.Linear(self.value_size * self.num_heads, self.hidden_size, bias=False)
        self.dropout = torch.nn.Dropout(config.attention_dropout)
        if config.use_route_values:
            self.Wroute = torch.nn.Linear(self.route_size, self.num_heads * self.value_size, bias=False)
        else:
            self.Wroute = None

    def forward(self, H, D, attention_mask):
        """
        Args:
          H   hiddens [batch, seq_len, hidden_size]
          D   route information [batch, 1, seq_len, route_size, seq_len]
          attention mask   0 if allowed -10000 for not allowed [batch, 1, 1, seq_len] or [batch, head, seq_len, seq_len]
        """
        H2_size = H.size()[:-1] + (self.num_heads, -1)
        H2 = self.W(H).view(*H2_size).transpose(1, 2)
        ## H2 is [batch, num_heads, seq_len, total_size]
        Q, K, V, R = torch.split(H2, [self.key_size, self.key_size, self.value_size, self.route_size], dim=-1)

        ## R is [batch, num_heads, n, route_size]:
        ## R => [batch, num_heads, n,  1, route_size]
        ## D is [batch, 1, n, route_size, n]
        RD      = torch.matmul(R.unsqueeze(-2), D).squeeze(-2)
        scale   = 1.0 / math.sqrt(self.key_size + self.route_size)
        attention_scores = scale * (Q @ K.transpose(-1, -2) + RD) + attention_mask

        ## note dropout is after softmax, so some connections are fully dropped!
        attention_probs  = self.dropout(attention_scores.softmax(dim=-1))


        context = attention_probs @ V
        ## if needed compute route values:
        if self.Wroute is not None:
            ## WD is [batch, seq_len, seq_len, value_size]
            WD = self.Wroute(D.sequeeze(1).transpose(-1, -2))
            context = context + torch.einsum("bhij,bijv->bhiv", attention_probs, WD)

        context_size  = H.size()[:2] + (self.num_heads * self.value_size,)
        ## context is [batch, num_heads, n, value_size]
        ## context_layer will be [batch, n, num_heads*value_size]
        context_layer = context.transpose(1, 2).contiguous().view(*context_size)

        return context_layer


class GraphSelfAttention(torch.nn.Module):
    def __init__(self, config, input_size=None):
        """
        Args:
          h   number of heads
        """
        super().__init__()
        self.num_heads   = config.num_heads
        self.key_size    = config.key_size
        self.key_r_size  = config.key_r_size
        self.value_size  = config.value_size
        self.hidden_size = config.hidden_size
        if input_size is None:
            self.input_size = self.hidden_size
        else:
            self.input_size = input_size
        self.route_size  = config.route_size
        
        self.injective = config.injective

        total_size   = self.key_size + self.key_size + self.value_size + self.key_r_size
        self.W       = torch.nn.Linear(self.input_size, self.num_heads * total_size, bias=False)
        self.Wd      = torch.nn.Linear(self.route_size, self.num_heads * self.key_r_size, bias=False)
        #self.Wout    = torch.nn.Linear(self.value_size * self.num_heads, self.hidden_size, bias=False)
        self.dropout = torch.nn.Dropout(config.attention_dropout)
        if config.use_route_values:
            self.Wroute = torch.nn.Linear(self.route_size, self.num_heads * self.value_size, bias=False)
        else:
            self.Wroute = None

    def forward(self, H, route_data, attention_mask):
        """
        Args:
          H   hiddens [batch, seq_len, hidden_size]
          ###D   route information [batch, 1, seq_len, route_size, seq_len]
          route_data       route information [batch, seq_len, seq_len, route_size]
          attention mask   0 if allowed -10000 for not allowed [batch, 1, 1, seq_len] or [batch, head, seq_len, seq_len]
        """
        H2_size = H.size()[:-1] + (self.num_heads, -1)
        H2 = self.W(H).view(*H2_size).transpose(1, 2)
        ## H2 is [batch, num_heads, seq_len, total_size]
        Qn, Kn, V, Qr = torch.split(H2, [self.key_size, self.key_size, self.value_size, self.key_r_size], dim=-1)

        Kr_size = route_data.size()[:-1] + (self.num_heads, self.key_r_size)
        Kr = self.Wd(route_data).view(*Kr_size).permute(0, 3, 1, 4, 2)

        ## Qr is [batch, num_heads, n, key_r_size]:
        ## Qr => [batch, num_heads, n, 1, key_r_size]
        ## Kr is [batch, num_heads, n, key_r_size, n]
        QKr     = torch.matmul(Qr.unsqueeze(-2), Kr).squeeze(-2)
        scale   = 1.0 / math.sqrt(self.key_size + self.key_r_size)
        ## TODO: move attent_dropout into attention_mask
        attention_scores = scale * (Qn @ Kn.transpose(-1, -2) + QKr) + attention_mask

        ## note dropout is after softmax, so some connections are totally dropped!
        if self.injective:
            attention_probs  = self.dropout(attention_scores.sigmoid())
        else:
            attention_probs  = self.dropout(attention_scores.softmax(dim=-1))

        self.attention_probs = attention_probs #for visualization

        context = attention_probs @ V
        ## if required compute route values:
        if self.Wroute is not None:
            ## Vr is [batch, seq_len, seq_len, value_size]
            Vr_size = route_data.size()[:-1] + (self.num_heads, self.value_size)
            Vr      = self.Wroute(route_data).view(*Vr_size).permute(0, 3, 1, 2, 4)
            context = context + torch.einsum("bhij,bhijv->bhiv", attention_probs, Vr)

        context_size  = H.size()[:2] + (self.num_heads * self.value_size,)
        ## context is [batch, num_heads, n, value_size]
        ## context_layer will be [batch, n, num_heads*value_size]
        context_layer = context.transpose(1, 2).contiguous().view(*context_size)

        return context_layer


class GraphSelfOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear      = torch.nn.Linear(config.num_heads * config.value_size, config.hidden_size)
        self.dropout     = torch.nn.Dropout(config.hidden_dropout)
        self.dropchannel = torch.nn.Dropout2d(config.channel_dropout)
        self.layer_norm  = LayerNorm(config.hidden_size, eps=1e-12)
        self.residual    = config.residual

    def forward(self, H, input_tensor):
        H = self.linear(H)
        H = self.dropout(H)
        if self.dropchannel.p > 0.0:
            H = self.dropchannel(H.transpose(-1, -2)).transpose(-1, -2)

        if self.residual:
            return self.layer_norm(H) + input_tensor
        return self.layer_norm(H + input_tensor)

class GraphAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = GraphSelfAttention(config)
        self.self_output    = GraphSelfOutput(config)

    def forward(self, H, D, attention_mask):
        Hnew = self.self_attention(H, D, attention_mask)
        Hnew = self.self_output(Hnew, H)
        return Hnew


class GraphOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear      = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout     = torch.nn.Dropout(config.hidden_dropout)
        self.dropchannel = torch.nn.Dropout2d(config.channel_dropout)
        self.layer_norm  = LayerNorm(config.hidden_size, eps=1e-12)
        self.residual    = config.residual

    def forward(self, H, input_tensor):
        H = self.linear(H)
        H = self.dropout(H)
        if self.dropchannel.p > 0.0:
            H = self.dropchannel(H.transpose(-1, -2)).transpose(-1, -2)

        if self.residual:
            H = self.layer_norm(H) + input_tensor
        else:
            H = self.layer_norm(H + input_tensor)
        return H

def gelu(x):
    """Implementation of the gelu activation function."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GILayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GraphAttention(config)
        self.intermediate = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.intermediate_size),
            torch.nn.ReLU()
        )
        self.output = GraphOutput(config)

    def forward(self, H, D, attention_mask):
        attention_output    = self.attention(H, D, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output        = self.output(intermediate_output, attention_output)
        return layer_output

class GIEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.ModuleList([GILayer(config) for _ in range(config.num_layers)])

    def forward(self, H, D, logit_attention_mask):
        for layer in self.layers:
            H = layer(H, D, logit_attention_mask)
        return H

class GIPooler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = torch.nn.Tanh()

    def forward(self, H):
        ## first token is assumed to be a pooling token
        return self.act_fn(self.linear(H[:, 0]))

def set_channel_dropout(module, prob):
    if isinstance(module, torch.nn.Dropout2d):
        module.p = prob

class GraphInformer(torch.nn.Module):
    """
    Args:
        config    configuration for GraphInformer

    Args for calling the created module:
        node_ids        type of given node, 0 if missing. int tensor: [batch, node]
        node_features   node features. float tensor: [batch, node, node_feature]
        route_data      for route information. float tensor: [batch, node, node, route_size]
        attention_mask  1 if visible 0 otherwise. float tensor: [batch, node]
        dists           distance between nodes. float tensor: [batch, node, node]
                        used for locality masking

    Returns:
        layer_layer  last layer embeddings for each node
    """
    def __init__(self, config):
        super().__init__()
        self.config     = config
        self.embeddings = GIEmbeddings(config)
        if config.embed_routes:
            self.route_embeddings = GIRouteEmbeddings(config)
        else:
            self.route_embeddings = torch.nn.Sequential()
        self.encoder    = GIEncoder(config)
        ## registering head_radius to have Pytorch manage it
        if config.head_radius is not None:
            self.register_buffer("head_radius", torch.Tensor(config.head_radius))
        else:
            self.head_radius = None
        self.apply(self.init_gi_weights)

    def init_gi_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            #module.weight.data.fill_(1.0)
            ## changed for pure residual setup
            if module.init_norm is None:
                module.weight.data.fill_(self.config.init_norm)
            else:
                module.weight.data.fill_(module.init_norm)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, node_ids, node_features, route_data, attention_mask=None, dists=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(node_ids)

        assert node_ids.shape == node_features.shape[0:2]
        assert node_ids.shape == attention_mask.shape
        assert node_ids.shape == route_data.shape[0:2]
        assert node_ids.shape[1] == route_data.shape[2]

        ## logit_attention_mask is [batch, 1, 1, n]
        logit_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        logit_attention_mask = (1.0 - logit_attention_mask) * -10000.0
        if self.head_radius is not None:
            assert dists.shape[0:2] == node_ids.shape
            assert dists.shape[2]   == dists.shape[1]
            ## TODO: try temperature adjusted heads
            r = self.head_radius.view(1,-1, 1, 1)
            d_mask = (r + 1e-3 < dists.unsqueeze(1)).float() * -10000.0
            ## d_mask has shape: [batch, head, node, node]
            logit_attention_mask = logit_attention_mask + d_mask

        ## no need to change route shape any more
        ## route data [batch, 1, node, route_size, node]
        ##D = route_data.transpose(-2, -1).unsqueeze(1)

        embedding_out = self.embeddings(node_ids, node_features)
        route_emb     = self.route_embeddings(route_data)
        last_layer    = self.encoder(embedding_out, route_emb, logit_attention_mask)
        return last_layer


class GINodeRegression(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gi = GraphInformer(config)
        self.config = config
        ## TODO: maybe try dropping Tanh or adding layers
        self.regression_output = torch.nn.Sequential(
            #torch.nn.ReLU(),
            #torch.nn.Linear(config.hidden_size, config.regression_size),
            #torch.nn.Tanh(),
            #torch.nn.Linear(config.regression_size, 1),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_size, 1),
        )
        self.regression_output.apply(self.init_gi_weights)

    def init_gi_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.init_norm)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, node_ids, node_features, route_data, attention_mask=None, dists=None):
        last_layer = self.gi(node_ids, node_features, route_data=route_data, attention_mask=attention_mask, dists=dists)
        output = self.regression_output(last_layer)
        return output


class GraphPooler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooler_type = config.pooler_type
        if config.pooler_pre_layer:
            self.pre = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(config.final_dropout),
            )
        else:
            self.pre = torch.nn.ReLU()

        if config.pooler_post_layer:
            self.post = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(config.final_dropout),
                torch.nn.Linear(config.hidden_size, config.graph_num_labels),
            )
        else:
            self.post = torch.nn.Linear(config.hidden_size, config.graph_num_labels)

    def forward(self, hiddens, attention_mask):
        out     = self.pre(hiddens)
        masked  = out * attention_mask.unsqueeze(-1)

        if self.pooler_type == "mean":
            pooled = masked.sum(-2) / attention_mask.sum(1).unsqueeze(-1)
        elif self.pooler_type == "max":
            pooled = masked.max(-2)
        elif self.pooler_type == "sum":
            pooled = masked.sum(-2)
        else:
            raise ValueError(f"No pooler_type '{self.pooler_type}'.")

        final = self.post(pooled)
        return final


class GIGraphClassification(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gi = GraphInformer(config)
        self.config = config
        ## TODO: maybe try adding layers
        self.pooler_net = GraphPooler(config)
        self.pooler_net.apply(self.init_gi_weights)

    def init_gi_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.init_norm)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_channel_dropout(self, prob):
        self.apply(lambda m: set_channel_dropout(m, prob))

    def forward(self, node_ids, node_features, route_data, attention_mask=None, dists=None):
        last_layer = self.gi(node_ids, node_features, route_data=route_data, attention_mask=attention_mask, dists=dists)
        out = self.pooler_net(last_layer, attention_mask)
        return out


class OutputAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        total_size      = 1 + config.out_value_size
        self.W          = torch.nn.Linear(config.hidden_size, config.out_num_heads * total_size)
        self.num_heads  = config.out_num_heads
        self.value_size = config.out_value_size

    def forward(self, H, attention_mask):
        """
        Args:
            H               hiddens [batch, seq_len, hidden_size]
            attention_mask  1.0 if allowed, 0.0 for masked nodes [batch, seq_len]
        """
        H2_size = H.size()[:-1] + (self.num_heads, -1)
        H2      = self.W(H).view(*H2_size).transpose(1, 2)
        ## H2 is [batch, heads, seq_len, total_size]
        Q, V = torch.split(H2, [1, self.value_size], dim=-1)

        attention_logits = (attention_mask - 1.0) * 10000
        attention_logits = attention_logits.unsqueeze(1)
        attention_probs  = (Q.squeeze(-1) + attention_logits).softmax(dim=-1)
        out = attention_probs.unsqueeze(-2) @ V
        out = out.view(H.size(0), -1)
        return out


class GIGraphClassificationAttn(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gi = GraphInformer(config)
        self.config = config
        self.output_net = OutputAttention(config)

        out_size      = config.out_num_heads * config.out_value_size
        if config.out_hidden_size is None:
            self.graph_class_net = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Dropout(config.final_dropout),
                torch.nn.Linear(out_size, config.graph_num_labels),
            )
        else:
            self.graph_class_net = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Dropout(config.final_dropout),
                torch.nn.Linear(out_size, config.out_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(config.final_dropout),
                torch.nn.Linear(config.out_hidden_size, config.graph_num_labels),
            )
        self.output_net.apply(self.init_gi_weights)
        self.graph_class_net.apply(self.init_gi_weights)

    def init_gi_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.init_norm)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, node_ids, node_features, route_data, attention_mask=None, dists=None):
        last_layer = self.gi(node_ids, node_features, route_data=route_data, attention_mask=attention_mask, dists=dists)
        out     = self.output_net(last_layer, attention_mask)
        logits  = self.graph_class_net(out)
        return logits

