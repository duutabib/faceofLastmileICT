import pandas as pd

import matplotlib.pyplot as plt
from typing import Optional, Union
import matplotlib.gridspec as gridspec


class Visualizer:
    def __init__(self,  fit_dict: dict = None, **kwargs,  ):
        self._fit_dict = fit_dict

    def _get_model_data(self, model_key:str) -> tuple[float, float, pd.Series]:
        try:
            score, mse, y_pred = self._fit_dict[model_key]
            return score, mse, y_pred
        except (KeyError, TypeError):
            raise ValueError(f"Invalid fit_dict format for {model_key}")
            

    def _plot_model(self,ax, y_actual, y_pred, score, mse, title: str, color:str):
        ax.scatter(y_actual, y_pred, color=color)
        ax.axline((0, 0), slope=1, linestyle='--', color='red', linewidth=2)
        ax.set_xlabel("X instances in time", fontsize=20)
        ax.set_title(f"Discharge Coefficient", fontsize=20)
        ax.set_title(f"{title} (Score ={score:.3f}) (mse ={mse:.3f})", fontsize=20)


    # maybe implement helper for plot asthetics including (axis, labels)
    # not sure able about this in this alpha version
    def _plot_decorator(labels, ax, bounds, ):
        ...

    # need to figure out where I set this...
    TITLE = [
            "$f(x, y) = a x^2 + b xy + cx+ dy+ ey^2 + f$: y-actual vs y_pred",
            "DecisionTreeRegressor: y-actual vs y_pred", 
            "actual model: y-actual vs y_pred",
    ]

    def apply_model_visualizer(self, models, all: bool=True, color: str ='blue', figsize: tuple = (20, 20),  **kwargs) -> Optional[list]:
        # all param (bool) = control number of fits compared...

        """Visual model predictions against actual data."""

        """
            returns a plot of all model fits for data, with the default all flag. Working on
            additional flags to allow for flexibilty in comparison. 
            Args:
                color: string; set color for labels
`               figsize: tuple; set size for the fig
                all: bool; rendering comparison for all models 
                **kwargs:
        """

        models = ['lm_model', 'poly_model', 'dTree_model'] if all else models
        models  = models or self._fit_dict.keys() - {y_actual}
                    
        fig0 = plt.figure(figsize= figsize, constrained_layout=True)
        ncols=len(models)
        w1, w2 = 20, 1
        gspec=gridspec.GridSpec(ncols=ncols, nrows=1, figure=fig0, width_ratios=[w1, w2])
        y_actual = self.fit_dict.get('y_actual', pd.Series())

        for i, model_key in enumerate(models):
            score, mse, y_pred = _get_model_data(model=model_key)
            ax = fig.add_subplot(gspec[0, i])
            self._plot_model(ax, y_actual, y_pred, score, mse, model_key.captialize(), color)
        plt.show()
    
            

class ImageHandler:
    """ takes objects from the  visualizer, resize 
    and saves it one of two formats {png, pdf}.
    """
    def __init__(self, fig: plt.Figure, name: str) -> None:
        if not isinstance(figure, plt.Figure):
            raise TypeError("figure must be a matplotlib.pyplot.Figure")
        self._fig = fig
        self.name = name

    # resize image
    def resize(self, width: int, height:int) -> None:
        self._fig.set_size_inches(width, height)

    # save image as pdf
    def save(self, format: str ='png') -> None:
        self._fig.savefig(self.name, format=format, bbox_inches='tight')
