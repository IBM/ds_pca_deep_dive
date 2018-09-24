
## Deepdive into Principal Component Analysis (PCA).

This Code Pattern will guide you through how to use `Scikit Learn` and `Python` in IBM Watson Studio. The goal is to use a Jupyter notebook is to deep dive into Principal Component Analysis using the various dataset that are shipped with 'Scikit Learn'.

We will first give a short intuitive explanation for principal component analysis and why it makes sense. Then we will go deeper into the actual derivation of Principal Components using the principle of maximizing the total projected variances onto components. Once we have understood the theory and concept, we will dive deeper into the use cases and examples. We will consider four scenarios with examples.

* Dimension Reduction
* Visualization
* Noise Filtering
* As a pre-processor for ML algorithms.

In the end, we will summarize our discussion with various pointers to alternatives to PCA.

![](doc/source/images/architecture.png)

## Flow

1. Log into IBM Watson Studio service.
2. Start a notebook in Watson Studio.
4. Deep dive into intuition and theory of PCA.
5. Consider four scenarios with Examples:
5.1. Dimension Reduction
5.2. Visualization
5.3. Noise Filtering
5.4. As a pre-processor for ML algorithms.

## Included components

* [IBM Watson Studio](https://dataplatform.ibm.com): Analyze data using RStudio, Jupyter, and Python in a configured, collaborative environment that includes IBM value-adds, such as managed Spark.
* [Jupyter Notebook](http://jupyter.org/): An open source web application that allows you to create and share documents that contain live code, equations, visualizations, and explanatory text.

## Featured technologies

* [Data Science](https://medium.com/ibm-data-science-experience/): Systems and scientific methods to analyze structured and unstructured data in order to extract knowledge and insights.
* [Python](https://www.python.org/): Python is a programming language that lets you work more quickly and integrate your systems more effectively.
* [Scikit Learn](http://scikit-learn.org/stable/):  A Python library for providing efficient tools for data mining and machine learning.
* [Matplotlib](https://matplotlib.org/): A Python library integrating matplot for visualization.

# Steps

This Code Pattern consists of following activities:

* [Run a Jupyter notebook in the IBM Watson Studio](#run-a-jupyter-notebook-in-the-ibm-watson-studio).
* [Deep Dive into Principal Component Analysis](#deep-dive-into-principal-component-analysis).

## Run a Jupyter notebook in the IBM Watson Studio

1. [Sign up for the Watson Studio](#1-sign-up-for-the-watson-studio)
2. [Create a new Watson Studio project](#2-create-a-new-watson-studio-project)
3. [Create the notebook](#3-create-the-notebook)
4. [Run the notebook](#4-run-the-notebook)
5. [Save and Share](#5-save-and-share)

### 1. Sign up for the Watson Studio

Log in or sign up for IBM's [Watson Studio](https://dataplatform.ibm.com).

> Note: if you would prefer to skip the remaining Watson Studio set-up steps and just follow along by viewing the completed Notebook, simply:
> * View the completed [notebook](https://github.com/aloknsingh/ds_pca_deep_dive/blob/master/notebooks/deep_dive_pca.ipynb) and its outputs, as is.
> * While viewing the notebook, you can optionally download it to store for future use.
> * When complete, continue this code pattern by jumping ahead to the [Principal Component Analysis in Practice.](#principal-component-analysis-in-practice) section.

### 2. Create a new Watson Studio project

* Select the `New Project` option from the Watson Studio landing page and choose the `Data Science` option.

![](https://raw.githubusercontent.com/IBM/pattern-images/master/watson-studio/project_choices.png)

* To create a project in Watson Studio, give the project a name and either create a new `Cloud Object Storage` service or select an existing one from your IBM Cloud account.

![](https://raw.githubusercontent.com/IBM/pattern-images/master/watson-studio/new_project.png)

* Upon a successful project creation, you are taken to a dashboard view of your project. Take note of the `Assets` and `Settings` tabs, we'll be using them to associate our project with any external assets (datasets and notebooks) and any IBM cloud services.

![](https://raw.githubusercontent.com/IBM/pattern-images/master/watson-studio/project_dashboard.png)

### 3. Create the Notebook

* From the project dashboard view, click the `Assets` tab, click the `+ New notebook` button.

![](https://raw.githubusercontent.com/IBM/pattern-images/master/watson-studio/new_notebook.png)

* Give your notebook a name and select your desired runtime, in this case we'll be using python Runtime.

* Now select the `From URL` tab to specify the URL to the notebook in this repository.

![](https://raw.githubusercontent.com/IBM/pattern-images/master/watson-studio/notebook_with_url_spark.png)

* Enter this URL:

```
https://github.com/aloknsingh/ds_pca_deep_dive/blob/master/notebooks/deep_dive_pca.ipynb
```

* Click the `Create` button.

> Note: If queried for a Python version, select version `3.5`.

### 4. Run the notebook

When a notebook is executed, what is actually happening is that each code cell in
the notebook is executed, in order, from top to bottom.

Each code cell is selectable and is preceded by a tag in the left margin. The tag
format is `In [x]:`. Depending on the state of the notebook, the `x` can be:

* A blank, this indicates that the cell has never been executed.
* A number, this number represents the relative order this code step was executed.
* A `*`, this indicates that the cell is currently executing.

There are several ways to execute the code cells in your notebook:

* One cell at a time.
  * Select the cell, and then press the `Play` button in the toolbar.
* Batch mode, in sequential order.
  * From the `Cell` menu bar, there are several options available. For example, you
    can `Run All` cells in your notebook, or you can `Run All Below`, that will
    start executing from the first cell under the currently selected cell, and then
    continue executing all cells that follow.
* At a scheduled time.
  * Press the `Schedule` button located in the top right section of your notebook
    panel. Here you can schedule your notebook to be executed once at some future
    time, or repeatedly at your specified interval.

### 7. Save and Share

#### How to save your work:

Under the `File` menu, there are several ways to save your notebook:

* `Save` will simply save the current state of your notebook, without any version
  information.
* `Save Version` will save your current state of your notebook with a version tag
  that contains a date and time stamp. Up to 10 versions of your notebook can be
  saved, each one retrievable by selecting the `Revert To Version` menu item.

#### How to share your work:

You can share your notebook by selecting the `Share` button located in the top
right section of your notebook panel. The end result of this action will be a URL
link that will display a “read-only” version of your notebook. You have several
options to specify exactly what you want shared from your notebook:

* `Only text and output`: will remove all code cells from the notebook view.
* `All content excluding sensitive code cells`:  will remove any code cells
  that contain a *sensitive* tag. For example, `# @hidden_cell` is used to protect
  your credentials from being shared.
* `All content, including code`: displays the notebook as is.
* A variety of `download as` options are also available in the menu.

## Deep Dive into Principal Component Analysis.

### Principal Component Analysis (PCA) Intuition.

Through various real life examples, we give user intuition of PCA.

### PCA Mathemathical Formulation.

We give user mathemathical foundation and derive the key idea of PCA.

### Principal Component Analysis in Practice.
 
Once, We have seen, intuition and mathematical formulation of Principal Component Analysis. We explore PCA through various examples. We will be using Scikit-learn and matplotlib to dive deep into examples.

We will look into following four use cases:

1. PCA for Dimension Reduction.

2. PCA for Visualization and Better Insights.

3. PCA for Noise Filtering.

4. PCA as a Preprocessor for ML algorithms.

## Sample Output

The following screen-shot shows, derivation of PCA by maximizing total projected variances.
![](doc/source/images/pca_derivation.png)

The following screen-shot shows, how one can do simple classification using PCA.
![](doc/source/images/pca_classification.png)

The following screen-shots show, how to de-noise a noisy image using PCA.
![](doc/source/images/pca_noisy_input_image.png)
![](doc/source/images/pca_noisy_output_image.png)

Awesome job following along! Now go try and take this further or apply it to a different use case!

## Links

- Watson Studio: https://datascience.ibm.com/docs/content/analyze-data/creating-notebooks.html.
- Scikit Learn: http://scikit-learn.org/stable/
- Matplotlib: https://matplotlib.org/
- SeaBorn: https://seaborn.pydata.org

# Learn more

* **Data Analytics Code Patterns**: Enjoyed this Code Pattern? Check out our other [Data Analytics Code Patterns](https://developer.ibm.com/code/technologies/data-science/)
* **AI and Data Code Pattern Playlist**: Bookmark our [playlist](https://www.youtube.com/playlist?list=PLzUbsvIyrNfknNewObx5N7uGZ5FKH0Fde) with all of our Code Pattern videos
* **Data Science Experience**: Master the art of data science with IBM's [Data Science Experience](https://datascience.ibm.com/)

# License

[Apache 2.0](LICENSE)
