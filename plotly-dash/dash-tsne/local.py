# -*- coding: utf-8 -*-

"""
The Local version of the app.
"""

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


def input_field(title, state_id, state_value, state_max, state_min):
    """Takes as parameter the title, state, default value and range of an input field, and output a Div object with
    the given specifications."""
    return html.Div([
        html.P(title,
               style={
                   'display': 'inline-block',
                   'verticalAlign': 'mid',
                   'marginRight': '5px',
                   'margin-bottom': '0px',
                   'margin-top': '0px'
               }),

        html.Div([
            dcc.Input(
                id=state_id,
                type='number',
                value=state_value,
                max=state_max,
                min=state_min,
                size=7
            )
        ],
            style={
                'display': 'inline-block',
                'margin-top': '0px',
                'margin-bottom': '0px'
            }
        )
    ]
    )

# Generate the default scatter plot
tsne_df = pd.read_csv("demo_embeddings/data.csv", index_col=0)

data = []

for idx, val in tsne_df.groupby(tsne_df.index):
    idx = int(idx)
    scatter = go.Scatter3d(
        name=f"{idx}",
        x=val['x'],
        y=val['y'],
        z=val['z'],
        mode='markers',
        marker=dict(
            size=2.5,
            symbol='circle-dot'
        )
    )
    data.append(scatter)
# Layout for the t-SNE graph
tsne_layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

Heading = html.Div([
html.H1("t-SNE", style= {
                        "font-weight": "300","font-size" : "5rem",
                        "font-family": "inherit", "font-kerning" : "normal"
   })
],className="container",style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px",
                                 "margin-bottom": "2rem", "text-align": "center",
                                 "padding-top": "60px"
                                 })


local_layout = html.Div([
    # Main app
    Heading,
    html.Div([
        html.Div([
            # The graph
            html.Div(
            dcc.Graph(
                id='tsne-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '80vh',
                }
            ),
              style= {"align" : "right"
                      }
            )
        ],
            id="plot-div",
        )],
            style={
                'padding': 20,
                'margin': 5,
                'borderRadius': 5,
                # Remove possibility to select the text for better UX
                'user-select': 'none',
                '-moz-user-select': 'none',
                '-webkit-user-select': 'none',
                '-ms-user-select': 'none'
            }
        )
    ],
    className="container",
    style={
        'font-size': '1.5rem',
        "background-color": "beige", "margin-top": "30px",
        "box-shadow": "rgb(240, 240, 240) 0px 0px 0px 0px",
        "padding-left": "250px",
        "margin-left": "200px",
        "margin-top": "-70px"
}
)

def local_callbacks(app):
    def parse_content(contents, filename):
        """This function parses the raw content and the file names, and returns the dataframe containing the data, as well
        as the message displaying whether it was successfully parsed or not."""

        if contents is None:
            return None, ""

        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))

            else:
                return None, 'The file uploaded is invalid.'
        except Exception as e:
            print(e)
            return None, 'There was an error processing this file.'

        return df, f'{filename} successfully processed.'

    # Uploaded data --> Hidden Data Div
    @app.callback(Output('data-df-and-message', 'children'),
                  [Input('upload-data', 'contents'),
                   Input('upload-data', 'filename')])
    def parse_data(contents, filename):
        data_df, message = parse_content(contents, filename)

        if data_df is None:
            return [None, message]

        elif data_df.shape[1] < 3:
            return [None, f'The dimensions of {filename} are invalid.']

        return [data_df.to_json(orient="split"), message]

    # Uploaded labels --> Hidden Label div
    @app.callback(Output('label-df-and-message', 'children'),
                  [Input('upload-label', 'contents'),
                   Input('upload-label', 'filename')])
    def parse_label(contents, filename):
        label_df, message = parse_content(contents, filename)

        if label_df is None:
            return [None, message]

        elif label_df.shape[1] != 1:
            return [None, f'The dimensions of {filename} are invalid.']

        return [label_df.to_json(orient="split"), message]

    # Hidden Data Div --> Display upload status message (Data)
    @app.callback(Output('upload-data-message', 'children'),
                  [Input('data-df-and-message', 'children')])
    def output_upload_status_data(data):
        return data[1]

    # Hidden Label Div --> Display upload status message (Labels)
    @app.callback(Output('upload-label-message', 'children'),
                  [Input('label-df-and-message', 'children')])
    def output_upload_status_label(data):
        return data[1]

    # Button Click --> Update graph with states
    @app.callback(Output('plot-div', 'children'),
                  [Input('tsne-train-button', 'n_clicks')],
                  [State('perplexity-state', 'value'),
                   State('n-iter-state', 'value'),
                   State('lr-state', 'value'),
                   State('pca-state', 'value'),
                   State('data-df-and-message', 'children'),
                   State('label-df-and-message', 'children')
                   ])
    def update_graph(n_clicks, perplexity, n_iter, learning_rate, pca_dim, data_div, label_div):
        """Run the t-SNE algorithm upon clicking the training button"""

        error_message = None  # No error message at the beginning

        # Fix for startup POST
        if n_clicks <= 0 or data_div is None or label_div is None:
            global data
            kl_divergence, end_time = None, None

        else:
            # Extract the data dataframe and the labels dataframe from the divs. they are both the first child of the div,
            # and are serialized in json
            data_df = pd.read_json(data_div[0], orient="split")
            label_df = pd.read_json(label_div[0], orient="split")

            # Fix the range of possible values
            if n_iter > 1000:
                n_iter = 1000
            elif n_iter < 250:
                n_iter = 250

            if perplexity > 50:
                perplexity = 50
            elif perplexity < 5:
                perplexity = 5

            if learning_rate > 1000:
                learning_rate = 1000
            elif learning_rate < 10:
                learning_rate = 10

            if pca_dim > data_df.shape[1]:  # We limit the pca_dim to the dimensionality of the dataset
                pca_dim = data_df.shape[1]
            elif pca_dim < 3:
                pca_dim = 3

            # Start timer
            start_time = time.time()

            # Apply PCA on the data first
            pca = PCA(n_components=pca_dim)
            data_pca = pca.fit_transform(data_df)

            # Then, apply t-SNE with the input parameters
            tsne = TSNE(n_components=3,
                        perplexity=perplexity,
                        learning_rate=learning_rate,
                        n_iter=n_iter)

            try:
                data_tsne = tsne.fit_transform(data_pca)
                kl_divergence = tsne.kl_divergence_

                # Combine the reduced t-sne data with its label
                tsne_data_df = pd.DataFrame(data_tsne, columns=['x', 'y', 'z'])

                label_df.columns = ['label']

                combined_df = tsne_data_df.join(label_df)

                data = []

                # Group by the values of the label
                for idx, val in combined_df.groupby('label'):
                    scatter = go.Scatter3d(
                        name=idx,
                        x=val['x'],
                        y=val['y'],
                        z=val['z'],
                        mode='markers',
                        marker=dict(
                            size=2.5,
                            symbol='circle-dot'
                        )
                    )
                    data.append(scatter)

                end_time = time.time() - start_time

            # Catches Heroku server timeout
            except:
                error_message = "We were unable to train the t-SNE model due to timeout. Try to run it again, or to clone the repo and run the program locally."
                kl_divergence, end_time = None, None

        return [
            # Data about the graph
            html.Div([
                kl_divergence
            ],
                id="kl-divergence",
                style={'display': 'none'}
            ),

            html.Div([
                end_time
            ],
                id="end-time",
                style={'display': 'none'}
            ),

            html.Div([
                error_message
            ],
                id="error-message",
                style={'display': 'none'}
            ),

            # The graph
            dcc.Graph(
                id='tsne-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '80vh',
                },
            )
        ]

    # Updated graph --> Training status message
    @app.callback(Output('training-status-message', 'children'),
                  [Input('end-time', 'children'),
                   Input('kl-divergence', 'children')])
    def update_training_info(end_time, kl_divergence):
        # If an error message was output during the training.

        if end_time is None or kl_divergence is None or end_time[0] is None or kl_divergence[0] is None:
            return None
        else:
            end_time = end_time[0]
            kl_divergence = kl_divergence[0]

            return [
                html.P(f"t-SNE trained in {end_time:.2f} seconds.",
                       style={'margin-bottom': '0px'}),
                html.P(f"Final KL-Divergence: {kl_divergence:.2f}",
                       style={'margin-bottom': '0px'})
            ]

    @app.callback(Output('error-status-message', 'children'),
                  [Input('error-message', 'children')])
    def show_error_message(error_message):
        if error_message is not None:
            return [
                html.P(error_message[0])
            ]

        else:
            return []
