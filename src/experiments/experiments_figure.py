import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl

steps = [132, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
xticks = [132, 300, 400, 600, 800, 1000, 1200, 1400, 1600]
baseline_data = {
    "VS Rand": {
        "Elite-GA": {
            "Goal Diff": {
                "MPI": [69, 70, 69, 70, 69, 69, 69, 68, 67, 67, 66, 66, 66, 65, 65, 64],
                "SVM-SS": [16, 51, 55, 57, 59, 58, 59, 60, 60, 61, 59, 58, 56, 56, 57, 57],
                "OBL": [34, 45, 44, 42, 41, 40, 40, 42, 40, 36, 32, 36, 32, 27, 26, 24],
                "KAES": [47, 65, 64, 64, 64, 63, 64, 62, 64, 64, 63, 62, 59, 59, 59, 57]
            },
            r"Avg": {
                "MPI": [71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 70, 69, 69, 69, 69, 69],
                "SVM-SS": [47, 59, 68, 70, 71, 70, 70, 70, 71, 71, 70, 69, 68, 68, 69, 69],
                "OBL": [63, 69, 67, 68, 68, 68, 68, 66, 65, 65, 65, 64, 65, 64, 64, 64],
                "KAES": [62, 72, 72, 71, 70, 69, 69, 68, 68, 69, 67, 67, 67, 68, 69, 68]
            }
        },
        "BRKGA": {
            "Goal Diff": {
                "MPI": [-3, 23, 34, 38, 35, 31, 33, 30, 31, 28, 30, 28, 24, 24, 18, 16],
                "SVM-SS": [-58, -37, -12, -1, 6, 11, 11, 13, 14, 13, 15, 14, 16, 16, 15, 17],
                "OBL": [9, 3, 10, 5, 5, 8, 7, 7, 3, 5, 6, 7, 7, 5, 2, 5],
                "KAES": [-37, -17, 10, 16, 20, 16, 16, 15, 17, 16, 15, 16, 10, 11, 14, 12]
            },
            r"Avg": {
                "MPI": [32, 52, 59, 61, 59, 58, 59, 58, 56, 51, 49, 49, 50, 48, 49, 50],
                "SVM-SS": [9, 17, 29, 41, 45, 47, 47, 45, 50, 48, 48, 49, 46, 47, 49, 54],
                "OBL": [43, 43, 44, 48, 46, 44, 43, 41, 43, 40, 41, 35, 38, 38, 37, 36],
                "KAES": [17, 30, 40, 50, 50, 49, 46, 48, 50, 46, 50, 49, 49, 50, 48, 47]
            }
        }
    },
    "MPI VS": {
        "Elite-GA": {
            "Goal Diff": {
                "Rand": [69, 70, 69, 70, 69, 69, 69, 68, 67, 67, 66, 66, 66, 65, 65, 64],
                "SVM-SS": [52, 50, 46, 43, 42, 40, 41, 38, 35, 36, 33, 32, 28, 30, 29, 29],
                "OBL": [61, 64, 67, 67, 68, 64, 64, 63, 62, 62, 63, 61, 59, 60, 60, 59],
                "KAES": [33, 40, 35, 34, 35, 34, 34, 36, 32, 32, 31, 31, 28, 27, 25, 25]
            },
            r"Avg": {
                "Rand": [71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 70, 69, 69, 69, 69, 69],
                "SVM-SS": [61, 60, 60, 57, 56, 56, 56, 55, 53, 54, 54, 54, 54, 54, 54, 53],
                "OBL": [69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69],
                "KAES": [56, 59, 61, 59, 59, 55, 57, 58, 57, 57, 58, 56, 56, 55, 56, 54],
            }
        },
        "BRKGA": {
            "Goal Diff": {
                "Rand": [-3, 23, 34, 38, 35, 31, 33, 30, 31, 28, 30, 28, 24, 24, 18, 16],
                "SVM-SS": [50, 42, 33, 27, 20, 18, 18, 18, 11, 10, 10, 8, 6, 8, 4, 3],
                "OBL": [-6, 21, 21, 23, 21, 25, 27, 25, 23, 22, 21, 13, 13, 12, 10, 11],
                "KAES": [33, 26, 14, 13, 14, 13, 16, 12, 5, 5, 4, 3, 3, -1, 0, -2],
            },
            r"Avg": {
                "Rand": [32, 52, 59, 61, 59, 58, 59, 57, 56, 51, 49, 49, 50, 48, 49, 50],
                "SVM-SS": [60, 57, 55, 52, 52, 47, 48, 48, 47, 51, 46, 45, 42, 42, 40, 39],
                "OBL": [31, 48, 56, 56, 55, 53, 52, 52, 49, 50, 50, 49, 49, 48, 47, 47],
                "KAES": [56, 55, 50, 48, 47, 43, 45, 45, 47, 43, 42, 41, 41, 37, 37, 36],
            }
        }
    }
}

ablation_data = {
    "Gatenet Obj": {
        "Elite-GA": {
            "Goal Diff": {
                "Mean": [-4, -2, -4, 1, 1, 3, 2, 4, 4, 3, 2, 0, 1, 2, 0, -1],
                "Mean+Div": [20, 11, 6, 7, 7, 4, 6, 4, 4, 2, 1, 1, 1, 2, 0, 0],
                "Max+Div": [30, 35, 27, 28, 27, 28, 28, 27, 21, 22, 20, 17, 17, 17, 16, 11]
            },
            r"Avg": {
                "Mean": [35, 40, 32, 35, 35, 38, 40, 42, 39, 40, 40, 35, 37, 38, 38, 37],
                "Mean+Div": [50, 53, 51, 49, 49, 47, 48, 47, 46, 43, 43, 42, 41, 38, 40, 41],
                "Max+Div": [63, 62, 60, 57, 53, 54, 53, 52, 54, 50, 51, 47, 49, 48, 46, 45],
            }
        },
        "BRKGA": {
            "Goal Diff": {
                "Mean": [-3, -1, 3, 0, 2, 3, 2, 4, 1, 1, 3, 1, 2, 1, 1, 0],
                "Mean+Div": [17, 7, 6, 1, -1, 8, 9, 5, 2, -1, -3, -2, -2, -2, -3, -3],
                "Max+Div": [23, 19, 13, 6, 10, 8, 7, 8, 5, 2, 2, 0, -2, -1, 3, 3],
            },
            r"Avg": {
                "Mean": [33, 38, 40, 36, 40, 35, 39, 40, 38, 38, 35, 30, 30, 31, 30, 30],
                "Mean+Div": [44, 44, 39, 33, 36, 33, 34, 33, 35, 34, 30, 29, 33, 30, 30, 31],
                "Max+Div": [63, 54, 47, 44, 42, 42, 46, 43, 40, 37, 39, 37, 39, 34, 32, 32],
            }
        }
    },
    "Gatenet Inter": {
        "Elite-GA": {
            "Goal Diff": {
                "No Gatenet": [27, 28, 18, 18, 15, 11, 12, 8, 6, 8, 7, 6, 4, 4, 5, 4],
                "No Interpolation": [25, 43, 44, 43, 38, 35, 35, 36, 33, 33, 30, 30, 28, 32, 26, 26],
                "No Transfer": [59, 57, 53, 52, 51, 49, 50, 48, 48, 46, 45, 44, 39, 40, 38, 38],
            },
            r"Avg": {
                "No Gatenet": [63, 58, 61, 57, 58, 58, 58, 57, 55, 54, 54, 53, 50, 52, 48, 46],
                "No Interpolation": [56, 66, 66, 65, 67, 64, 61, 59, 61, 62, 61, 60, 59, 60, 59, 59],
                "No Transfer": [66, 65, 63, 64, 62, 61, 61, 60, 59, 59, 58, 59, 59, 59, 58, 58],
            }
        },
        "BRKGA": {
            "Goal Diff": {
                "No Gatenet": [22, 15, 8, 4, 6, 6, 5, 9, 9, 5, 8, 4, 7, 5, 3, 4],
                "No Interpolation": [-2, 11, 14, 19, 13, 14, 14, 13, 8, 10, 5, 7, 7, 4, 5, -1],
                "No Transfer": [55, 52, 39, 24, 18, 13, 15, 20, 15, 14, 13, 13, 10, 11, 9, 8],
            },
            r"Avg": {
                "No Gatenet": [52, 54, 50, 50, 44, 43, 45, 47, 42, 41, 39, 35, 36, 36, 37, 38],
                "No Interpolation": [34, 51, 47, 43, 45, 50, 50, 46, 43, 40, 37, 40, 38, 39, 39, 40],
                "No Transfer": [66, 64, 55, 51, 47, 46, 43, 46, 41, 42, 40, 41, 38, 40, 40, 39],
            }
        }
    }
}

if __name__ == '__main__':
    colors = ["#81271f", "#006335", "#57c3c2", "#00a0e9", "#e4007f", "#35272a", "#0d559a"]
    # color_list = [[colors[i] for i in [0, 2, 3, 4]], [colors[i] for i in [1, 2, 3, 4]]]
    color_list = [[colors[i] for i in [1, 2, 3, ]], [colors[i] for i in [1, 2, 3, ]]]
    for exp_idx, exp in enumerate(ablation_data.keys()):
        exp_data = ablation_data[exp]
        for ga in exp_data.keys():
            ga_data = exp_data[ga]
            for indicate in ga_data.keys():
                indicate_data = ga_data[indicate]
                plot_dict = {
                    r"\#FEs": [],
                    indicate: [],
                    "Method": []
                }
                for method in indicate_data.keys():
                    for idx, record in enumerate(indicate_data[method]):
                        plot_dict[r"\#FEs"].append(steps[idx])
                        plot_dict[indicate].append(record)
                        plot_dict["Method"].append(method)
                df = pd.DataFrame(plot_dict)
                plt.figure(figsize=(4, 4))
                plt.clf()
                mpl.rcParams['pgf.texsystem'] = 'pdflatex'
                mpl.rcParams['pgf.rcfonts'] = False
                min_y, max_y = min(plot_dict[indicate]), max(plot_dict[indicate])
                hor_line = 0 if indicate == "Goal Diff" else 36
                if hor_line - min_y < 15:
                    plt.ylim(hor_line - 15, max_y + 2)
                plt.xlim(100, 1650)
                # plt.ylim(-10 if indicate == "Goal Diff" else 0, 72)
                plt.axhline(y=hor_line, color='#FF0000', linestyle='-.',
                            linewidth=1.5)
                ax = sns.lineplot(df, x=r"\#FEs", y=indicate, hue="Method",
                                  style="Method", markers=True, linewidth='1',
                                  palette=color_list[exp_idx])
                ax.fill()
                ax.fill_betweenx([hor_line, ax.get_ylim()[1]], 100, 1650, color='red',
                                 alpha=0.3)
                ax.fill_betweenx([ax.get_ylim()[0], hor_line], 100, 1650, color='green',
                                 alpha=0.3)
                plt.xticks(ticks=xticks, rotation=60)
                plt.title(f"{indicate} on {ga}")
                plt.tight_layout()
                # plt.show()
                plt.savefig(f"../../figs/{exp}_{ga}_{indicate}.pdf", backend='pgf')
