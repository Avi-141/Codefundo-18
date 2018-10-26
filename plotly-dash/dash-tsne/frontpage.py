import base64
import io
import os
import time

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


front_layout = html.Div([])