import graphinformer as gi
import torch

ga = gi.MHGraphAttention1(din=32, dk=8, dv=8, dr=15, h=4, dmodel=32, dropout=0.1)

H = torch.randn(10, 32)
D = torch.randn(10, 10, 15)
attention_mask = torch.zeros(10, 10)

H2 = ga(H, D, attention_mask)


#### Batched version
config = gi.GIConfig(
        num_heads         = 4,
        key_size          = 8,
        key_r_size        = 6,
        value_size        = 8,
        hidden_size       = 32,
        intermediate_size = 100,
        route_size        = 7,
        node_feature_size = 8,
        num_node_types    = 12,
        attention_dropout = 0.1,
        hidden_dropout    = 0.1,
        num_layers        = 8,
        initializer_range = 0.1,
        embedding_dropout = 0.)
gsa = gi.GraphSelfAttention(config)

H1   = torch.randn(16, 22, 32)
D    = torch.randn(16, 22, 22, 7)
mask = torch.zeros(16, 1, 1, 22)

H2 = gsa(H1, D, mask)

####
gilayer = gi.GILayer(config)
H2 = gilayer(H1, D, mask)

giencoder = gi.GIEncoder(config)
H8 = giencoder(H1, D, mask)

#### embeddings
node_types    = torch.randint(12, size=(16, 22))
node_features = torch.randn(16, 22, 8)
giemb = gi.GIEmbeddings(config)
emb   = giemb(node_types, node_features)

#### GraphInformer ####
route_data  = torch.randn(16, 22, 22, 7)
attn        = torch.ones(16, 22)
model       = gi.GraphInformer(config)
out = model(
    node_ids       = node_types,
    node_features  = node_features,
    route_data     = route_data,
    attention_mask = attn
)


#### Test locality priority #####
config2 = gi.GIConfig(
        num_heads         = 4,
        key_size          = 8,
        key_r_size        = 6,
        value_size        = 8,
        hidden_size       = 32,
        intermediate_size = 100,
        route_size        = 7,
        node_feature_size = 8,
        num_node_types    = 12,
        attention_dropout = 0.1,
        hidden_dropout    = 0.1,
        num_layers        = 8,
        initializer_range = 0.1,
        head_radius       = [2.0, 2.0, 4.0, 999.0],
        embedding_dropout = 0.1,
)
dists = torch.randn(16, 22, 22)
model2 = gi.GraphInformer(config)

out = model2(
    node_ids       = node_types,
    node_features  = node_features,
    route_data     = route_data,
    attention_mask = attn,
    dists          = dists,
)

#### Test route values (G3) #####
config2 = gi.GIConfig(
        num_heads         = 4,
        key_size          = 8,
        key_r_size        = 6,
        value_size        = 8,
        hidden_size       = 32,
        intermediate_size = 100,
        route_size        = 7,
        node_feature_size = 8,
        num_node_types    = 12,
        attention_dropout = 0.1,
        hidden_dropout    = 0.1,
        num_layers        = 8,
        initializer_range = 0.1,
        head_radius       = [2.0, 2.0, 4.0, 999.0],
        use_route_values  = True,
        embedding_dropout = 0.1,
)
dists3 = torch.randn(16, 22, 22)
model3 = gi.GraphInformer(config2)

out = model3(
    node_ids       = node_types,
    node_features  = node_features,
    route_data     = route_data,
    attention_mask = attn,
    dists          = dists3,
)

