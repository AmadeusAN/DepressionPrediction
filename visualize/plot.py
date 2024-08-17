import numpy as np
import plot
import pandas as pd
from os.path import join
import plotly.express as px

LOSS_DIR = r"/public1/cjh/workspace/DepressionPrediction/visualize/train/CSENet"
loss_name = ["mse_loss", "rmse_loss"]
target_epoch = 100

def merge_loss():
    

def show_line_chart():
    loss_list = []
    for loss in loss_name:
        loss_np = np.load(join(LOSS_DIR, loss + "_epoch_to_" + str(target_epoch) + ".npz"))[
            "data"
        ]
        loss_list.append(loss_np)

    loss_list_np = np.stack(loss_list, axis=-1)
    df = pd.DataFrame(loss_list_np, columns=loss_name)
    fig = px.line(df, x=df.index, y=loss_name, title="test plot")
    fig.show()
