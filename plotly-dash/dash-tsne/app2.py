# -*- coding: utf-8 -*-
import os
import dash
import base64
import dash_core_components as dcc
import dash_html_components as html
import plotly.offline
from local import local_layout
app = dash.Dash(__name__, assets_external_path= "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js")
server = app.server
# Load external CSS
external_css = [
"https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
    "//fonts.googleapis.com/css?family=Raleway:400,300,600",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    "https://cdn.rawgit.com/plotly/dash-tsne/master/custom_styles.css",
    "https://cdn.rawgit.com/plotly/dash-app-stylesheets/2cc54b8c03f4126569a3440aae611bbef1d7a5dd/stylesheet.css",
]

for css in external_css:
    app.css.append_css({"external_url": css})
set = "set0"
image1 = '/assets/'+set+'/en.jpg'
image2 = "/assets/"+set+"/img.jpg"
image3 = "/assets/"+set+"/pred.jpg"
header1_css ={
"margin-top": "90px",
"margin-bottom": "2rem",
"font-weight": "300",
"font-size" : "9rem",
"font-family": "inherit",
"font-kerning" : "normal"
}

para1 = """Variational autoencoders (VAEs) are a deep learning technique for learning latent representations. They have also been used to draw images, achieve state-of-the-art results in semi-supervised learning, as well as interpolate between sentences."""
para2 = """VAEs try to reconstruct output from input and consist of an encoder and a decoder, which are encoding and decoding the data. The encoder outputs a compressed representation of the output data. The decoder then learns to reconstruct the initial input data by taking this compressed representation as input data. This way, you can use the decoder as a generative model which is able to generate specific features — such as specific digits or letters."""
para3 = """We have used Convolutional VAE to compress and encode the time series dataset of everyday cloud patterns from Archived cyclone images. We load realtime satellite data and use encoder to encode the image. """
para4 = """One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task, such as using previous video frames might inform the understanding of the present frame. If RNNs could do this, they’d be extremely useful.
Sometimes, we only need to look at recent information to perform the present task. For example, consider a language model trying to predict the next word based on the previous ones. If we are trying to predict the last word in “the clouds are in the sky,” we don’t need any further context – it’s pretty obvious the next word is going to be sky. In such cases, where the gap between the relevant information and the place that it’s needed is small, RNNs can learn to use the past information.
We use RNN with LSTM cells to predict next time steps encoded form by using encoded forms of previous images. This forecasted encoded vector is then reconstructed using our decoder which was trained in the VAE.
We use the similarity between latent representations to represent the probability of a cyclone forming. Similarity is mathematically defined as the inner product of encoded vectors 
"""
Imgblock = html.Div([
            html.Img(src=image1, className="left-block",
                     style = {"padding-left":"330px",
                              "height": "370px"
                              }),
            html.P("Encoded form:", className = "left-block",
                   style={"padding-left": "330px",
                          }
                   ),
            ],className="container",style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px",
                                        })
Imgblock2 =   html.Div([html.Img(src = image2, className = "center-block",
                      style={
                          "padding-left":"95px"
                      }),
    html.Img(src=image3, className="right-block",
             style={
                 "padding-left": "95px",
                 "width" : "400px"
             })
             ],className="container",style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px",
                                        })
paragraph2 = html.Div([html.P(para2,style={
                         "padding-left":"30px",
                          "font-size" :"2.0rem"
                 })
                       ],className="container",style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px",
                })
paragraph3 = html.Div([html.P(para3,style={
                         "padding-left":"30px",
                          "font-size" :"2.0rem"
                 })
                       ],className="container",style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px",
                })

paragraph4 = html.Div([html.P(para4,style={
                         "padding-left":"30px",
                          "font-size" :"2.0rem"
                 })
                       ],className="container",style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px",
                })

lstmHeading = html.Div([
html.H1("LSTM", style= {
                        "font-weight": "300","font-size" : "5rem",
                        "font-family": "inherit", "font-kerning" : "normal"
   })
],className="container",style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px",
                                 "margin-bottom": "2rem", "text-align": "center",
                                 "padding-top": "60px"
                                 })

tsneGraph = html.Div(
              [local_layout],
            style ={
                "background-color": "beige", "margin-top": "30px",
                "box-shadow": "rgb(240, 240, 240) 0px 0px 0px 0px",
            }
)
labels = html.Div([
                    html.P("Input to encoder", className = "center-block",
                           style={
                            "padding-left": "155px",
                            "margin-top": "-50px"
                           }),
                    html.P("Decoded image", className  = "right-block",
                           style={
                               "padding-left": "570px",
                               "margin-top": "-50px"
                           })
],className="container",style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px",
                                        })

caption = html.Div([

],className="container",style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px",
                                        })
app.layout = html.Div(children=[
    html.Div([
    html.Nav([html.A("Project Toofan",
                    className = "navbar-brand text-white",
                    style={
                        "text-align" : "left",
                        "font-size" : "medium",
                        "letter-spacing":"3px",
                        "font-style" : "italic"
                    }
   )],className ="navbar fixed-top",style={"background-color":"#414344"})],className="container-fluid"),
   html.Div([
   html.Div([html.H1("VAE", style= {
                        "font-weight": "300","font-size" : "5rem",
                        "font-family": "inherit", "font-kerning" : "normal"
   }),
             ],className = "container",
            style = { "background-color" : "beige",
                     "margin-bottom": "2rem","text-align" : "center",
                       "padding-top" : "60px","box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px"
   })],className="container-fluid", style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px"}),
    html.Div([ html.P(
                 para1,
                 style={
                         "padding-left":"30px",
                          "font-size" :"2.0rem"
                 }),
          ],className="container",style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px",
                                        }),
             Imgblock,
             paragraph2,
             Imgblock2,
             labels,
             paragraph3,
             lstmHeading,
             paragraph4,
             tsneGraph
],className="container-fluid", style = {"background-color" : "beige","margin-top": "30px",
                                       "box-shadow" : "rgb(240, 240, 240) 0px 0px 0px 0px",
                                       })
# Running the server
app.run_server(debug=True)

