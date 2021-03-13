from torch.utils.data import DataLoader
import graphinformer as gi
import numpy as np
import nmr_loader
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR

from tensorboardX import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='NMR prediction.')
parser.add_argument('-d', '--data_file', help='Data file', type=str, default="./nmr_webo.npy")
parser.add_argument('-f', '--folding_file', help='Data file', type=str, default="./folding0.npy")
parser.add_argument('-r', '--head_radius', nargs='+', help='Head radius', type=int, default=[1, 1, 1, 1, 1, 1])
parser.add_argument('-i', '--init_norm', help='Initial weight for LayerNorm', type=float, default=0.01)
parser.add_argument('--init_norm_emb', help='Initial weight for Emb. LayerNorm', type=float, default=0.01)
parser.add_argument('-l', '--layers', help='Number of layers', type=int, default=8)
parser.add_argument('--hidden_size', help='Hidden size', type=int, default=384)
parser.add_argument('--attention_dropout', help='Attention dropout', type=float, default=0.0)
parser.add_argument('--final_dropout', help='Final dropout', type=float, default=0.0)
parser.add_argument('--hidden_dropout', help='Hidden dropout', type=float, default=0.0)
parser.add_argument('--embedding_dropout', help='Embedding dropout', type=float, default=0.0)
parser.add_argument('--channel_dropout', help='Channel dropout', type=float, default=0.0)
parser.add_argument('--weight_decay', help='Weight decay', type=float, default=0.0)
parser.add_argument('-n', '--name', help="Name of run", type=str, default=None)
parser.add_argument('-b', '--batch_size', help="Batch size", type=int, default=64)
parser.add_argument('--learning_rate', help="Learning rate", type=float, default=1e-3)
parser.add_argument('--num_epochs', help="Number of epochs", type=int, default=100)
parser.add_argument('--lr_steps', help="Epochs to reduce learning rate", type=int, nargs='+', default=[50, 80])
parser.add_argument('--lr_gamma', help="Learning rate reduction rate", type=float, default=0.3)
parser.add_argument('--use_route_values', action="store_true")
parser.add_argument('--residual', help="Use pure residual", type=int, default=1)
parser.add_argument('--embed_routes', action="store_true")
parser.add_argument('--no_bar', help="Disables progress bar", action="store_true")
args = parser.parse_args()
args.save = True #Save best model

if args.name is None:
    ## automatic title
    args.name = (
        f"gi{args.layers}"
        f"_{args.hidden_size}"
        f"_h{''.join([str(i) for i in args.head_radius])}"
        f"_do{args.hidden_dropout}-{args.attention_dropout}-{args.final_dropout}-{args.channel_dropout}"
        f"_res{int(args.residual)}"
        f"_rv{int(args.use_route_values)}"
        f"_wd{args.weight_decay}"
        f"_b{args.batch_size}"
    )
    if "webo" in args.data_file:
        args.name += "_webo"

writer = SummaryWriter("runs/" + args.name)

print(args)
print("Run name: " + args.name)

num_epochs = args.num_epochs
dev = "cuda:0"

loader_tr, loader_va, loader_te = nmr_loader.loaders(
        data_file    = args.data_file,
        folding_file = args.folding_file,
        batch_size   = args.batch_size,
)

if args.embed_routes:
    route_size = 32
else:
    route_size = loader_tr.dataset.route_size

num_heads = len(args.head_radius)
key_size  = args.hidden_size // num_heads

config = gi.GIConfig(
    num_heads         = num_heads,
    key_size          = key_size,
    key_r_size        = key_size,
    value_size        = key_size,
    hidden_size       = args.hidden_size,
    intermediate_size = args.hidden_size * 2,
    route_size        = route_size,
    route_feature_size = loader_tr.dataset.route_size,
    node_feature_size = loader_tr.dataset.node_feature_size,
    num_node_types    = max(loader_tr.dataset.node_types.values()) + 1,
    attention_dropout = args.attention_dropout,
    hidden_dropout    = args.hidden_dropout,
    embedding_dropout = args.embedding_dropout,
    final_dropout     = args.final_dropout,
    channel_dropout   = args.channel_dropout,
    num_layers        = args.layers,
    initializer_range = 0.1,
    head_radius       = args.head_radius,
    init_norm         = args.init_norm,
    init_norm_emb     = args.init_norm_emb,
    use_route_values  = args.use_route_values,
    regression_loss   = "L1",
    regression_size   = 128,
    embed_routes      = args.embed_routes,
    weight_decay      = args.weight_decay,
    learning_rate     = args.learning_rate,
    residual          = args.residual > 0,
)
net = gi.GINodeRegression(config).to(dev)
if config.regression_loss == "L1":
    loss = torch.nn.L1Loss(reduction="none")
elif config.regression_loss == "L2":
    loss = torch.nn.MSELoss(reduction="none")
else:
    raise ValueError("Unknown regression_loss.")

def evaluate(loader):
    ## evaluate data in loader
    net.eval()
    lsum   = 0.0
    lcount = 0
    with torch.no_grad():
        for b in loader:
            node_ids = b["node_ids"].to(dev)
            mask = (node_ids >= 1).float()
            pred = net(
                node_ids,
                node_features  = b["node_features"].to(dev),
                route_data     = b["route_features"].to(dev),
                attention_mask = mask,
                dists          = b["dists"].to(dev),
            )
            pred        = pred.squeeze(-1)[:,1:]
            node_labels = b["node_labels"].to(dev)
            b_loss      = loss(pred, node_labels)
            loss_mask   = (1.0 - torch.isnan(node_labels).float()) * mask[:,1:]
            losses      = b_loss[loss_mask == 1.0]

            lsum   += losses.sum().cpu().item()
            lcount += losses.shape[0]
    return lsum / lcount


def report_performance(epoch, live_mae_tr, best):
    mae_tr = evaluate(loader_tr)
    mae_va = evaluate(loader_va)
    mae_te = evaluate(loader_te)
    writer.add_scalar(f"regression/mae_tr", mae_tr, epoch)
    writer.add_scalar(f"regression/mae_va", mae_va, epoch)
    writer.add_scalar(f"regression/mae_te", mae_te, epoch)
    is_best = False
    if mae_va < best["mae_va"]:
        best["mae_tr"] = mae_tr
        best["mae_va"] = mae_va
        best["mae_te"] = mae_te
        best["epoch"]  = epoch
        is_best = True

    print(f"{epoch}. mae_tr={mae_tr:.5f} ({live_mae_tr:.5f}), mae_va={mae_va:.5f}, mae_te={mae_te:.5f}")
    return is_best

optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

best = {"mae_va": np.inf}

for epoch in range(num_epochs):
    net.train()

    loss_sum   = 0.0
    loss_count = 0

    for b in tqdm.tqdm(loader_tr, disable=args.no_bar):
        optimizer.zero_grad()
        node_ids = b["node_ids"].to(dev)
        mask = (node_ids >= 1).float()
        ## DEBUGGING: switching off partial charges
        #b["node_features"][:,:,20] = 0.0
        pred = net(
            node_ids,
            node_features  = b["node_features"].to(dev),
            route_data     = b["route_features"].to(dev),
            attention_mask = mask,
            dists          = b["dists"].to(dev),
        )
        pred        = pred.squeeze(-1)[:,1:]
        node_labels = b["node_labels"].to(dev)
        b_loss      = loss(pred, node_labels)
        loss_mask   = (1.0 - torch.isnan(node_labels).float()) * mask[:,1:]
        losses      = b_loss[loss_mask == 1.0]
        final_loss  = losses.sum() / mask.shape[0]
        final_loss.backward()

        optimizer.step()

        loss_sum   += losses.detach().sum().cpu()
        loss_count += losses.shape[0]

    scheduler.step()
    loss_tr = loss_sum.item() / loss_count
    is_best = report_performance(epoch, loss_tr, best=best)
    if is_best and args.save:
        torch.save(net.state_dict(), f"runs/{args.name}_best_epoch")

writer.close()
np.save(f"runs/{args.name}.run.npy", {
    "config": config.__dict__,
    "best": best,
    "param_count": gi.count_parameters(net),
    "num_epochs": num_epochs,
})
print(f"Best: Epoch {epoch}. mae_tr={best['mae_tr']:.5f}, mae_va={best['mae_va']:.5f}, mae_te={best['mae_te']:.5f}")
