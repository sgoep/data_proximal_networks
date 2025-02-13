import os


def generate_latex_figure_block(image_filenames, captions, labels, figures_per_row=4):
    """
    Generates LaTeX code for a figure environment with subfigures.

    :param image_filenames: List of image filenames to include as subfigures.
    :param figures_per_row: Number of subfigures per row.
    :return: A string containing the LaTeX code.
    """

    print(image_filenames)
    # print(captions)
    print(labels)
    latex_code = "\\begin{figure}[b]\n"
    latex_code += "\\centering\n"
    latex_code += "\\setlength{\\figwidth}{0.24\\textwidth}\n"

    for i, image_file in enumerate(image_filenames):
        if i % figures_per_row == 0 and i != 0:
            latex_code += "\\\\ \n"  # Start a new line after every row of figures

        # Create a subfigure block for each image
        latex_code += "  \\begin{subfigure}{\\figwidth}\n"
        latex_code += (
            f"    \\includegraphics[width=\\textwidth]{{images/{image_file}}}\n"
        )
        latex_code += f"    \\caption{{{captions[i]}}}\n"
        latex_code += f"    \\label{{{labels[i]}}}\n"
        latex_code += "  \\end{subfigure}\n"

    latex_code += "\\end{figure}\n"

    return latex_code
