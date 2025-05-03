# ---------------- IMPORTS--------------------------
from turtle import width
from fastapi import FastAPI, HTTPException , Request
from fastapi.responses import HTMLResponse
import plotly.express as px
from plotly.io import to_html
import pandas as pd
import seaborn as sns
from fastapi.templating import Jinja2Templates
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
import io
import uvicorn
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx



#------------- Config-----------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
host = "127.0.0.1"
port = 8099

# ------------ Functions and Logic -------------------

# Load data
df = pd.read_csv('../BostonHousing.csv')

# Creating the target column
# Feature weights and desirability calculation (as in your original code)
features_for_quality = ['lstat', 'rm', 'nox', 'dis', 'ptratio', 'crim']
feature_weights = {'lstat': -0.74, 'rm': 0.70, 'nox': -0.43, 'dis': 0.25, 'ptratio': -0.51, 'crim': -0.39}

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features_for_quality])
scaled_df = pd.DataFrame(scaled_features, columns=features_for_quality)

for col in feature_weights:
    scaled_df[col] = scaled_df[col] * feature_weights[col]

scaled_df['desirability_score'] = scaled_df.sum(axis=1)
threshold = scaled_df['desirability_score'].median()
df['desirability'] = (scaled_df['desirability_score'] > threshold).astype(int)

correlation_with_desirability = df.corr()['desirability'].abs().sort_values(ascending=False)

# 3- 


# ---------- Routes --------------------------
@app.get('/')
async def root():
  return {'test':'done'}


@app.get('/Analysis')
async def analysis(request: Request):

    # Data to Dict to show it in the website 
    data = df.to_dict(orient="records")
    headers = df.columns.tolist()

    # Generate descriptive statistics
    df_description = df.describe().reset_index().to_dict(orient="records")
    description_headers = df.describe().reset_index().columns.tolist()

    # Calc Correlation 
    correlation = df.corr()




    # Plots 
    fig_corr = px.imshow(
        correlation,
        text_auto='.2f',
        color_continuous_scale='Blues',
        aspect='auto',
        title='Feature Correlation Heatmap'
    )

    fig_corr.update_layout(
        width=900,
        height=800,
        coloraxis_colorbar=dict(title='Correlation'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )


    
    # dists
    fig_dists = make_subplots(rows=5, cols=3, subplot_titles=df.columns[:-1])  # exclude 'desirability'

    row = 1
    col = 1
    for feature in df.columns[:-1]:
        fig_dists.add_trace(
            go.Histogram(
                x=df[feature],
                name=feature,
                marker_color='#1f77b4',
                opacity=0.7
            ),
            row=row, col=col
        )
        
        # Update grid position
        col += 1
        if col > 3:
            col = 1
            row += 1

    fig_dists.update_layout(
        title_text='Distribution of All Features',
        height=900,
        width=800,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))  
    )

    # Scatter pair plots 

    key_features = ['lstat', 'rm', 'nox', 'ptratio', 'medv', 'desirability']
    pairplot_df = df[key_features].copy()
    pairplot_df['desirability'] = pairplot_df['desirability'].astype(str)  # Convert to categorical for coloring

    fig_scatter = px.scatter_matrix(
        pairplot_df,
        dimensions=key_features[:-1],
        color='desirability',
        color_discrete_map={'0': 'white', '1': 'blue'},
        title='Pairplot of Key Features Colored by Desirability',
        width=1200,
        height=1200
    )

    fig_scatter.update_traces(
        diagonal_visible=False,
        showupperhalf=False,
        marker=dict(size=3, opacity=0.5)    
    )
    fig_scatter.update_layout(
    height=900,
    width=800,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),  
    title_font=dict(color='white'),  
    xaxis=dict(tickfont=dict(color='white')), 
    yaxis=dict(tickfont=dict(color='white'))  
    )

    # para 
    # Select features for parallel coordinates
    parallel_features = ['lstat', 'rm', 'nox', 'ptratio', 'crim', 'medv', 'desirability']

    fig_para = px.parallel_coordinates(
        df[parallel_features],
        color='desirability',
        color_continuous_scale=['white', 'blue'],
        dimensions=parallel_features[:-1],
        title='Parallel Coordinates Plot of Key Features'
    )

    fig_para.update_layout(
        height=700,
        width=1200,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))  
    )

    # radar

    radar_features = ['lstat', 'rm', 'nox', 'ptratio', 'crim', 'dis']
    df_radar = df.groupby('desirability')[radar_features].mean().reset_index()

    fig_radar = go.Figure()

    for i in range(len(df_radar)):
        fig_radar.add_trace(go.Scatterpolar(
            r=df_radar.loc[i, radar_features].values,
            theta=radar_features,
            fill='toself',
            name=f"Desirability {df_radar.loc[i, 'desirability']}",
            line_color='purple' if df_radar.loc[i, 'desirability'] == 0 else 'blue'
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[2, 10]
            )),
        showlegend=True,
        title='Radar Chart Comparing Feature Means by Desirability',
        width=700,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))  
    )

    # mix
    fig_mix = px.scatter(
        df,
        x='lstat',
        y='rm',
        color='desirability',
        color_discrete_map={0: 'purple', 1: 'blue'},  # Change dot colors
        marginal_x='histogram',
        marginal_y='histogram',
        title='Interactive Scatter Plot with Marginal Distributions',
        hover_data=['medv', 'nox', 'ptratio']
    )

    fig_mix.update_traces(
        marker=dict(size=8, opacity=0.6),
        selector=dict(mode='markers')
    )

    fig_mix.update_layout(
        width=800,
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # violin
    
    top_correlated = correlation_with_desirability.index[1:7]

    fig_violin = make_subplots(rows=2, cols=3, subplot_titles=top_correlated)

    for i, feature in enumerate(top_correlated):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        fig_violin.add_trace(
            go.Violin(
                x=df['desirability'][df['desirability'] == 0],
                y=df[feature][df['desirability'] == 0],
                name='Low Desirability',
                box_visible=True,
                meanline_visible=True,
                fillcolor='white',
                opacity=0.6,
                line_color='white',
                showlegend=(i == 0)
            ),
            row=row, col=col
        )
        
        fig_violin.add_trace(
            go.Violin(
                x=df['desirability'][df['desirability'] == 1],
                y=df[feature][df['desirability'] == 1],
                name='High Desirability',
                box_visible=True,
                meanline_visible=True,
                fillcolor='blue',
                opacity=0.6,
                line_color='blue',
                showlegend=(i == 0)
            ),
            row=row, col=col
        )

    fig_violin.update_layout(
        title='Violin Plots of Top Features by Desirability',
        height=600,
        width=900,
        violingap=0,
        violinmode='overlay',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # contor
    fig_contor = px.density_contour(
    df,
    x='lstat',
    y='rm',
    color='desirability',
    color_discrete_map={0: 'white', 1: 'blue'},
    title='Density Contour Plots by Desirability',
    width=800,
    height=600
    )

    fig_contor.update_traces(
        contours_coloring="fill",
        contours_showlabels=True,
        selector=dict(type='histogram2dcontour')
    )
    fig_contor.update_layout(
        height=600,
        width=900,
        violingap=0,
        violinmode='overlay',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )


    # bar
    mean_values = df.groupby('desirability')[top_correlated].mean().T.reset_index()
    mean_values = mean_values.melt(id_vars='index', var_name='desirability', value_name='mean_value')
    mean_values['desirability'] = mean_values['desirability'].map({0: 'Low', 1: 'High'})

    fig_bar = px.bar(
        mean_values,
        x='index',
        y='mean_value',
        color='desirability',
        barmode='group',
        color_discrete_map={'Low': 'white', 'High': 'blue'},
        title='Mean Feature Values by Desirability Level',
        labels={'index': 'Feature', 'mean_value': 'Mean Value'},
        width=800,
        height=500
    )

    fig_bar.update_layout(xaxis_tickangle=-45,       
        width=800,
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
                          
                          
    )

    # hexben

    fig_hex = go.Figure()

    fig_hex.add_trace(go.Histogram2dContour(
        x=df['lstat'],
        y=df['rm'],
        colorscale='Hot',
        reversescale=True,
        xaxis='x',
        yaxis='y',
        colorbar=dict(title='Point Density')
    ))

    fig_hex.add_trace(go.Scatter(
        x=df['lstat'],
        y=df['rm'],
        xaxis='x',
        yaxis='y',
        mode='markers',
        marker=dict(
            color='rgba(0,0,0,0.3)',
            size=3
        ),
        showlegend=False
    ))

    fig_hex.add_trace(go.Histogram(
        y=df['rm'],
        xaxis='x2',
        marker=dict(color='rgba(0,0,0,1)')
    ))

    fig_hex.add_trace(go.Histogram(
        x=df['lstat'],
        yaxis='y2',
        marker=dict(color='rgba(0,0,0,1)')
    ))

    fig_hex.update_layout(
        autosize=False,
        xaxis=dict(
            zeroline=False,
            domain=[0,0.85],
            showgrid=False
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0,0.85],
            showgrid=False
        ),
        xaxis2=dict(
            zeroline=False,
            domain=[0.85,1],
            showgrid=False
        ),
        yaxis2=dict(
            zeroline=False,
            domain=[0.85,1],
            showgrid=False
        ),
        height=600,
        width=800,
        bargap=0,
        hovermode='closest',
        showlegend=False,
        title='Hexbin Plot with Marginal Distributions (LSTAT vs RM)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
    )

    # rag
    features_to_animate = ['lstat', 'rm', 'nox', 'ptratio', 'crim']
    animation_df = df.melt(id_vars=['medv', 'desirability'], 
                        value_vars=features_to_animate,
                        var_name='feature', 
                        value_name='value')

    fig_reg = px.scatter(
        animation_df,
        x='value',
        y='medv',
        color='desirability',
        color_discrete_map={0: 'red', 1: 'blue'},
        animation_frame='feature',
        range_x=[animation_df['value'].min(), animation_df['value'].max()],
        range_y=[animation_df['medv'].min(), animation_df['medv'].max()],
        title='Animated Scatter Plots with Regression Lines',
        trendline='lowess',
        width=800,
        height=600
    )

    fig_reg.update_layout(
        xaxis_title='Feature Value',
        yaxis_title='Median Home Value (MEDV)',
        width=800,
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # network

    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Create graph
    G = nx.Graph()

    # Add nodes
    for feature in corr_matrix.columns:
        G.add_node(feature, size=10)

    # Add edges based on correlation
    threshold = 0.5
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                G.add_edge(
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    weight=corr_matrix.iloc[i, j]
                )

    # Create Plotly figure
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(15 + 10 * corr_matrix[node].mean())

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=[corr_matrix[node]['medv'] for node in G.nodes()],
            colorbar=dict(
                thickness=15,
                title='Correlation with MEDV',
                xanchor='left',
                titleside='right'
            ),
            line_width=2)
    )

    fig_network = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Network Graph of Feature Relationships',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    width=800,
                    height=600,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )
    fig_network.update_layout(
        width=800,
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # bar2 
    avg_df = df.groupby('desirability').mean().reset_index()

    fig_bar2 = go.Figure()

    for col in df.columns[:-1]:
        fig_bar2.add_trace(go.Bar(
            x=['Low', 'High'],
            y=avg_df[col],
            name=col
        ))

    fig_bar2.update_layout(
        barmode='group',
        title='Average Feature Values by Desirability',
        xaxis_title='Desirability',
        yaxis_title='Average Value',
        height=600,
        width=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )
    
    # cor
    correlation = df.corr()['desirability'].drop('desirability').sort_values()

    fig_cor = go.Figure()
    fig_cor.add_trace(go.Scatter(
        x=correlation.index,
        y=correlation.values,
        mode='markers+lines',
        line=dict(color='blue'),
        marker=dict(size=12, color=correlation.values, colorscale='RdBu'),
        name='Correlation'
    ))
    fig_cor.update_layout(
        title='Correlation of Features with Desirability',
        yaxis_title='Correlation',
        xaxis_title='Feature',
        height=600,
        width=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # corrr
    target_corr = df.corr()['medv'].drop('medv').sort_values()

    fig_corrr = px.bar(
        target_corr,
        title="Correlation of Each Feature with Median House Value (medv)",
        labels={'value': 'Correlation', 'index': 'Feature'},
        color=target_corr.values,
        color_continuous_scale='Viridis'
    )
    fig_corrr.update_traces(marker_line_width=1, marker_line_color="black")

    fig_corrr.update_layout(
        height=600,
        width=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )
    
    # bubble 
    fig_bubble = px.scatter(df, x='lstat', y='rm', color='desirability', size='medv',
                 hover_data=['crim', 'tax', 'ptratio'],
                 color_discrete_map={0: 'red', 1: 'blue'},
                 title='Interactive Hover: lstat vs rm with medv bubble size')
    fig_bubble.update_layout(
        height=600,
        width=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # bar3 

    features = ['crim', 'rm', 'ptratio', 'lstat', 'tax']
    grouped_means = df.groupby('desirability')[features].mean().T
    grouped_means.columns = ['Low Desirability', 'High Desirability']
    grouped_means = grouped_means.reset_index().rename(columns={'index': 'Feature'})

    fig_bar3 = go.Figure(data=[
        go.Bar(name='Low Desirability', x=grouped_means['Feature'], y=grouped_means['Low Desirability'], marker_color='red'),
        go.Bar(name='High Desirability', x=grouped_means['Feature'], y=grouped_means['High Desirability'], marker_color='blue')
    ])

    fig_bar3.update_layout(
        barmode='group',
        title='Grouped Bar Chart: Feature Means by Desirability',
        xaxis_title='Feature',
        yaxis_title='Average Value',
        height=600,
        width=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # river
    chas_counts = df.groupby(['chas', 'desirability']).size().reset_index(name='count')
    chas_counts['chas'] = chas_counts['chas'].map({0: 'Not Near River', 1: 'Near River'})
    chas_counts['desirability'] = chas_counts['desirability'].map({0: 'Low', 1: 'High'})

    fig_river = px.bar(
        chas_counts, x='chas', y='count', color='desirability',
        title='Desirability Distribution: Near River vs. Not',
        labels={'chas': 'Charles River Dummy', 'count': 'Count'},
        barmode='group',
        color_discrete_map={'Low': 'red', 'High': 'blue'}
    )

    fig_river.update_layout(
        height=600,
        width=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # rad
    rad_price = df.groupby('rad')['medv'].median().reset_index()

    fig_rad = px.bar(
        rad_price, x='rad', y='medv',
        title='Median House Price vs Accessibility to Highways (rad)',
        labels={'rad': 'Accessibility to Highways', 'medv': 'Median House Value ($1000s)'},
        color='medv',
        color_continuous_scale='Viridis'
    )
    fig_rad.update_layout(
        height=700,
        width=900,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # trend
    fig_trend = px.scatter(
    scaled_df, x='desirability_score', y=df['medv'],
    trendline='ols',
    title='Desirability Score vs. Median House Price',
    labels={'desirability_score': 'Neighborhood Desirability Score', 'y': 'Median Value ($1000s)'},
    color=df['desirability'].map({0: 'Low', 1: 'High'}),
    color_discrete_map={'Low': 'red', 'High': 'blue'}
    )
    fig_trend.update_layout(
        height=700,
        width=900,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # hist
    fig_hist = px.histogram(df, x='medv', nbins=30, color='desirability',
                   title='Distribution of Median Home Value by Desirability',
                   barmode='overlay', opacity=0.6,
                   color_discrete_map={0: 'red', 1: 'blue'})

    fig_hist.update_layout(
        height=700,
        width=900,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # 3d 
    fig_3d = px.scatter_3d(
    df, x='rm', y='lstat', z='medv',
    color='desirability',
    color_continuous_scale='Viridis',
    title='3D View: Rooms, Lower Status %, and House Value',
    labels={'rm': 'Rooms', 'lstat': 'Lower Status %', 'medv': 'House Price'}
    )

    fig_3d.update_layout(
        height=700,
        width=900,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )

    # hw
    heatmap_data = df.groupby(['rad', 'ptratio'])['medv'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='rad', columns='ptratio', values='medv')

    fig_hw = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='YlGnBu',
        colorbar=dict(title='Avg House Price')
    ))

    fig_hw.update_layout(
        title='Heatmap: Avg House Price by Highway Access (rad) & PTRATIO',
        xaxis_title='PTRATIO',
        yaxis_title='RAD (Accessibility to Highways)',
        height=700,
        width=900,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis=dict(tickfont=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white'))
    )



    
    #-----------------------------------------------------------------------------

    # convert the plots to html  

    corr_div = to_html(fig_corr, full_html=False)
    dists_div = to_html(fig_dists, full_html=False)
    scatter_div = to_html(fig_scatter, full_html=False)
    para_div = to_html(fig_para, full_html=False)
    radar_div = to_html(fig_radar, full_html=False)
    mix_div = to_html(fig_mix, full_html=False)
    violin_div = to_html(fig_violin, full_html=False)
    contor_div = to_html(fig_contor, full_html=False)
    bar_div = to_html(fig_bar, full_html=False)
    hex_div = to_html(fig_hex, full_html=False)
    reg_div = to_html(fig_reg, full_html=False)
    network_div = to_html(fig_network, full_html=False)
    bar2_div = to_html(fig_bar2, full_html=False)
    cor_div = to_html(fig_cor, full_html=False)
    corrr_div = to_html(fig_corrr, full_html=False)
    bubble_div = to_html(fig_bubble, full_html=False)
    bar3_div = to_html(fig_bar3, full_html=False)
    river_div = to_html(fig_river, full_html=False)
    rad_div = to_html(fig_rad, full_html=False)
    trend_div = to_html(fig_trend, full_html=False)
    hist_div = to_html(fig_hist, full_html=False)
    f3d_div = to_html(fig_3d, full_html=False)
    hw_div = to_html(fig_hw, full_html=False)

    


    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "headers": headers,
        "data": data,
        "description_headers": description_headers,
        "df_description": df_description,
        'corr_div': corr_div,
        'dists_div': dists_div,
        'scatter_div':scatter_div,
        'para_div':para_div,
        'radar_div':radar_div,
        'mix_div':mix_div,
        'violin_div':violin_div,
        'contor_div':contor_div,
        'bar_div':bar_div,
        'hex_div':hex_div,
        'reg_div':reg_div,
        'network_div':network_div,
        'bar2_div':bar2_div,
        'cor_div':cor_div,
        'corrr_div':corrr_div,
        'bubble_div':bubble_div,
        'bar3_div':bar3_div,
        'river_div':river_div,
        'rad_div':rad_div,
        'trend_div':trend_div,
        'hist_div':hist_div,
        'f3d_div':f3d_div,
        'hw_div':hw_div
    })


# Run App
if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8074
    uvicorn.run("main:app", host=host, port=port, reload=True)