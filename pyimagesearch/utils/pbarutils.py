import progressbar


def build_widgets(msg, sep=' '):
    widgets = [msg, progressbar.Percentage(), sep, progressbar.Bar(), sep, progressbar.ETA()]
    return widgets
