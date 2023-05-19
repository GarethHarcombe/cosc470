# Gareth 06/03/2023 - Data Visualisation and other tools

from matplotlib import cm
import matplotlib.pyplot as plt

MARGIN = 30


def plot_map(points, laps, pc="blue", lc="red", cmap='brg'):
    """
    Inputs: 
        points and laps to be plotted
        pc: colour of normal points. Default blue, can specify variable like "Timestamp" to colour by time
        lc: colour of lap points. Default red, yellow also good if using brg
        cmap: used if there is a range of colours. gray and brg useful gradients

    prints map of points, with options for colouring

    Output: Returns none, but displays plot
    """

    max_diff = max(points.x.max() - points.x.min(), 
                   points.y.max() - points.y.min())
    
    ax = points.plot.scatter(x="x", 
                             y="y",
                             c=pc,
                             s=5,
                             xlim=(points.x.min() - MARGIN, max(points.x.max(), points.x.min() + max_diff) + MARGIN),
                             ylim=(points.y.min() - MARGIN, max(points.y.max(), points.y.min() + max_diff) + MARGIN),
                             cmap=cm.get_cmap(cmap),
                             figsize=(7, 7)
    )

    laps.plot.scatter(x="end_x",
                      y="end_y",
                      alpha=0.7,
                      c=lc,
                      ax=ax)
    
    ax.legend(["Data points", "Laps"])
