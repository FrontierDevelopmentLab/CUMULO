import numpy as np

VISDOMWINDOWS = {}

def line_plot(viz, title, x, y):
    if title in VISDOMWINDOWS:
        window = VISDOMWINDOWS[title]
        viz.line(X=[x], Y=[y], win=window, update='append', opts={'title': title})
    else:
        window = viz.line(X=[x], Y=[y], opts={'title': title})
        VISDOMWINDOWS[title] = window

def line_plot_per_dim(viz, title, x, y):

    for i, single_y in enumerate(y):
        line_plot(viz, title + " class {}".format(i), x, single_y)

def line_plot_mean(viz, title, x, y):

    line_plot(viz, "mean " + title, x, np.mean(y))


def scatter_plot(viz, title, x):
    if title in VISDOMWINDOWS:
        window = VISDOMWINDOWS[title]
        viz.scatter(X=x, win=window, update='replace', opts={'title': title})
    else:
        window = viz.scatter(X=x, opts={'title': title})
        VISDOMWINDOWS[title] = window


def images_plot(viz, title, x):
    if title in VISDOMWINDOWS:
        window = VISDOMWINDOWS[title]
        viz.images(x, win=window, opts={'title': title})
    else:
        window = viz.images(x, opts={'caption': title})
        VISDOMWINDOWS[title] = window
