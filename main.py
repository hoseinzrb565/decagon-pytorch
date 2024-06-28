from decagon.data import DecagonDataset, split_graph
from decagon.train import train

# *Configurations*
DATA_DIR = "./data"
FORCE_RELOAD = False
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
DIMS = [128, 64, 32]
DROPOUT_PS = [0.1, 0.1, 0.1]
DEVICE = "cpu"
EPOCHS = 200
OPTIMIZER = "adamw"
LR = 0.01


# *Load and split positive and negative graphs*
print("Loading dataset...")
dataset = DecagonDataset(
    root_dir=DATA_DIR,
    force_reload=FORCE_RELOAD,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
)
pos_g, neg_g = dataset[0].to(DEVICE), dataset[1].to(DEVICE)

train_pos_g, val_pos_g, test_pos_g = split_graph(g=pos_g)
train_neg_g, val_neg_g, test_neg_g = split_graph(g=neg_g)

# *Begin training*
print("Begin training...")
train(
    train_pos_g=train_pos_g,
    train_neg_g=train_neg_g,
    val_pos_g=val_pos_g,
    val_neg_g=val_neg_g,
    device=DEVICE,
    dims=DIMS,
    dropout_ps=DROPOUT_PS,
    epochs=EPOCHS,
    optimizer_name=OPTIMIZER,
    lr=LR,
)
