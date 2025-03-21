from typing import Optional, Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class DataVisualizer:
    def __init__(self,  fit_dict: dict = None, **kwargs,  ):
        self._fit_dict = fit_dict

    def _get_model_data(self, model_key:str) -> tuple[float, float, pd.Series]:
        try:
            score, mse, y_pred = self._fit_dict[model_key]
            return score, mse, y_pred
        except (KeyError, TypeError):
            raise ValueError(f"Invalid fit_dict format for {model_key}")
            

    def _plot_model(self,ax, y_actual, y_pred, score, title: str, color:str):
        ax.scatter(y_actual, y_pred, color=color)
        ax.axline((0, 0), slope=1, linestyle'--', color='red', linewidth=2)
        ax.set_title(f"{title} (Score ={score:.3f})", fontsize)


    # maybe implement helper for plot asthetics including (axis, labels)
    def _plot_decorator(labels, ax, bounds, ):
        ...

    def apply_model_visualizer(self, models, color: str ='blue', figsize: tuple = (20, 20), all: bool, **kwargs) -> Optional[list]:
            # Implementation: how to handle compare for specific type of models different 
            # need to think a bit about the design... 
            # create helper functions for each visualization of a model compose for the compare
            # that might work...

            """
                returns a plot of all model fits for data, with the default all flag. Working on
                additional flags to allow for flexibilty in comparison. 
                Args:
                    color: string; set color for labels
                    figsize: tuple; set size for the fig
                    all: bool; rendering comparison for all models 
                    **kwargs:
            """

            models = ['lm_model', 'poly_model', 'dTree_model'] if all else []
            models  = models or self._fit_dict.keys() - {y_actual}
                        
            # define data (fix outputs for model_mse)
            _, _, y_actual = _get_model_data(model='y_actual')
            lm_score, lm_mse, ypred_lm = _get_model_data(model='lm_model')
            _, poly_mse, poly_ypred = _get_model_data(model='poly_model')
            dTree_score, dTree_mse,  ypred_dTree = _get_model_data(model='dTree_model')
            

            fig0 = plt.figure(figsize= figsize, constrained_layout=True)
            ncols= 1, len(models)
            w1, w2 = 20, 1
            gspec=gridspec.GridSpec(ncols=ncols, nrows=1, figure=fig0, width_ratios=[w1, w2]))

            # 
            ax = plt.subplot(gspec[0, 0], aspect=1)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 4 + 1))
            ax.set_xlabel("X instances in time")
            ax.set_ylim(0, 1)
            ax.set_yticks(np.linspace(0, 1, 4 + 1))
            ax.set_ylabel("Discharge Coefficient", fontsize=20)
            ax.set_title("Title", family="Roboto", weight=500)
            ax.axline((0.00, 0.00), slope=1.0, linestyle='--', color='red', linewidth=2.0)
            ax.set_title(f"actual model: y-actual vs y_pred ({ lm_score=:.3f}) {lm_mse=:.3f}", fontsize=20)
            I = ax.scatter(y_actual, y_lm, color=color)

            # Decision tree compare
            ax = plt.subplot(gspec[0, 1], aspect=1)
            ax.set_xlabel("X instances in time", fontsize=20)
            ax.set_title(f"DecisionTreeRegressor: y-actual vs y_pred ({dTree_score=:.3f} {dTree_mse=:.3f})", fontsize=20)
            ax.axline((0.0, 0.0), slope=1, linestyle='--', color='red', linewidth=2.0)
            ax.scatter(y_actual, y_dTree, color=color)

            # poly plot 
            ax = plt.subplot(gspec[0, 2], aspect=1)
            ax.set_xlabel("X instances in time", fontsize=20)
            ax.axline((0.0, 0.0), slope=1, linestyle='--', color='red', linewidth=2.0)
            ax.set_title(f"$f(x, y) = a x^2 + b xy + cx+ dy+ ey^2 + f$: y-actual vs y_pred (R_sq={None} {poly_mse=:.3f})", fontsize=20)
            I = ax.scatter(y_actual, y_poly, color=color)
        return I


# takes objects from the  visualizer, resize and saves it one of two formats {png, pdf}.
class ImageHandler:
    def __init__(self, fig: plt.Figure, name: str) -> None:
        if not isinstance(figure, plt.Figure):
            raise TypeError("figure must be a matplotlib.pyplot.Figure")
        self._fig = fig
        self.name = name

    # resize image
    def _resize_image(self, width: int, height:int) -> None:
        self._fig.set_size_inches(width, height)

    # save image as pdf
    def save(self, format: str ='png') -> None:
        self._fig.savefig(self.name, format=format, bbox_inches='tight')
