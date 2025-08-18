# The code below was inspired by the Churnometer Project from Code Institute 
# with some adjustments
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.data_management import load_breast_cancer_data, load_pkl_file


def page_cluster_body():

    # load cluster analysis files and pipeline
    version = 'v1'

    cluster_pipe = load_pkl_file(
        f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl")
    cluster_silhouette = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_silhouette.png")
    features_to_cluster = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/features_define_cluster.png")
    cluster_profile = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv")
    cluster_features = (pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
                        .columns
                        .to_list()
                        )

    # dataframe for cluster_distribution_per_variable()
    df_diagnosis_vs_clusters = load_breast_cancer_data().filter(['diagnosis'], axis=1)
    df_diagnosis_vs_clusters['Clusters'] = cluster_pipe['model'].labels_

    st.write("### ML Pipeline: Cluster Analysis")

    # display pipeline training summary conclusions
    st.info(
        f"* We refitted the cluster pipeline using fewer variables, and it delivered equivalent "
        f"performance to the pipeline fitted using all variables.\n"
        f"* The pipeline average silhouette score is 0.49"
    )
    st.write("---")

    st.write("#### Cluster ML Pipeline steps")
    st.write(cluster_pipe)

    st.write("#### The features the model was trained with")
    st.write(cluster_features)

    st.write("#### Clusters Silhouette Plot")
    st.image(cluster_silhouette)

    cluster_distribution_per_variable(
        df=df_diagnosis_vs_clusters, target='diagnosis')

    st.write("#### Most important features to define a cluster")
    st.image(features_to_cluster)

    # text based on "06 - Modeling and Evaluation - Cluster Sklearn" 
    # notebook conclusions
    st.write("#### Cluster Profile")

    # hack to not display the index in st.table() or st.write()
    cluster_profile.index = pd.Index([" "] * len(cluster_profile))
    st.table(cluster_profile)

    statement = (
        f"Since we have set '0' as Benign (B) and '1' as Malignant (M), from "
        f"the above profiling we can describe each clusters in the following -\n\n"
        f"* Cluster 0 shows smooth tumor contours with low concavity (0.021–0.06) "
        f"and compact perimeters (79.85–99.29). The worst area area values "
        f"(470.2–708.3) indicate smaller tumor sizes, while the low fractal dimension "
        f"(0.069–0.083) reflects minimal structural chaos. Diagnoses are predominantly "
        f"benign (90%), with only 10% malignant, aligning with its low-risk nature.\n\n"
        f"* Cluster 1 is characterized by very irregular tumor margins (concavity 0.133–0.22) "
        f"and large tumor sizes (worst area 1436.8–2009.3). "
        f"Perimeter values (145.2–171.3) highlight aggressive growth, "
        f"while fractal dimensions (0.076–0.093) show moderate disorder. "
        f"Notably, 100% of the diagnoses in this cluster are malignant (M), "
        f"consistent with a high-risk profile.\n\n"
        f"* Cluster 2 lies between the two extremes, with moderately irregular "
        f"shapes (concavity 0.114–0.201) and mid-sized tumors "
        f"(area 639.2–981.1, perimeter 99.18–122.95). "
        f"However, the fractal dimension (0.103–0.124) is the highest "
        f"among clusters, reflecting greater structural chaos. "
        f"This cluster is mixed but leans malignant (72% malignant vs. 28% benign), "
        f"suggesting cases that may represent early malignancies or borderline "
        f"conditions requiring further evaluation."
    )
    st.info(statement)

    statement_short = (
        f"A concise summary of the clusters:\n\n"
        f"* Cluster 0: Mostly benign (90%), with smooth contours, smaller "
        f"tumor sizes, and low structural disorder.\n"
        f"* Cluster 1: Entirely malignant (100%), marked by large tumor "
        f"size, irregular margins, and aggressive growth.\n"
        f"* Cluster 2: Mixed (72% malignant), with moderate size, "
        f"irregular shapes, and the highest structural chaos."
    )
    st.success(statement_short)

# code coped from "06 - Modeling and Evaluation - Cluster Sklearn" notebook - 
# under "Cluster Analysis" section


def cluster_distribution_per_variable(df, target):

    df_bar_plot = df.groupby(
        ['Clusters', target]).size().reset_index(name='Count')
    df_bar_plot.columns = ['Clusters', target, 'Count']
    df_bar_plot[target] = df_bar_plot[target].astype('object')

    print(f"Clusters distribution across {target} levels")
    fig = px.bar(df_bar_plot, x='Clusters', y='Count',
                 color=target, width=800, height=500)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    fig.show(renderer='jupyterlab')

    df_relative = (df
                   .groupby(["Clusters", target])
                   .size()
                   .unstack(fill_value=0)
                   .apply(lambda x: 100 * x / x.sum(), axis=1)
                   .stack()
                   .reset_index(name='Relative Percentage (%)')
                   .sort_values(by=['Clusters', target])
                   )

    print(f"Relative Percentage (%) of {target} in each cluster")
    fig = px.line(df_relative, x='Clusters', y='Relative Percentage (%)',
                  color=target, width=800, height=500)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    fig.update_traces(mode='markers+lines')
    st.plotly_chart(fig)
