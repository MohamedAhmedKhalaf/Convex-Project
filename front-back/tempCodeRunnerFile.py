    fig_dist1 = go.Figure(data=[go.Histogram(x=df['crim'], nbinsx=30)])
    fig_dist1.update_layout(
        title_text="Distribution of crim", xaxis_title="crim", yaxis_title="Count", width=600, height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  
        title_font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white')), 
        yaxis=dict(tickfont=dict(color='white'))  

        
        )