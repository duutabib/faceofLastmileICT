import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


@pd.api.extensions.register_dataframe_accessor("visualizer")
class DataVisualizer:
    def __init__(self, fitDict: dict, **kwargs,  ):
        self.fitDict = fitDict

    def apply_model_visualizer(self, color: str ='blue', figsize: tuple = (20, 20), all: bool, **kwargs) -> :
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
            fitDict = self.fitDict

            p=plt.rcParams
            p["figure.figsize"] = figsize
            p["font.sans-serif"] = ["Roboto Condensed"]
            p["font.weight"] = "light"
            p["ytick.minor.visible"] = "True"
            p["xtick.minor.visible"] = "True"

            # define data (fix outputs for model_mse)
            y_actual = fitDict['y_actual']
            lm_score, y_lm =  fitDict['lm_model']
            poly_mse, y_poly =fitDict['poly_model'][1] 
            dTree_score, y_dTree = fitDict['dTree_model'][1]
            

            fig0 = plt.figure(constrainted_layout=True)
            nrows, ncols= 1, 3
            w1, w2 = 20, 1
            gspec=gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig0, width_ratios=[w1, w2]))

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
@pd.api.extensions.register_dataframe_accessor("ImageHandler")
class ImageHandler:
    def __init__(self, img, name) -> None:
        self.img = img 
        self.name = name

    # resize image
    def _resize_image(self, width, height) -> None:
        self.img = plt.gca()
        self.img.set_size_inches(width, height)

    # save image as pdf
    def save_pdf(self) -> None:
        self.img.savefig(f"{self.name}",
                        format="pdf", bbox_inches="tight")

    # save image as png 
    def save_png(self) - None:
        self.img.savefig(f"{self.name}", 
                        format="png", bbox_inches="tight")