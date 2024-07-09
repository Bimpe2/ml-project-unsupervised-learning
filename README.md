# machine_learning_project-unsupervised-learning

## Project Outcomes
- Unsupervised Learning: perform unsupervised learning techniques on a wholesale data dataset. The project involves four main parts: exploratory data analysis and pre-processing, KMeans clustering, hierarchical clustering, and PCA.
### Duration:
Approximately 1 hour and 40 minutes
### Project Description:
In this project, we will apply unsupervised learning techniques to a real-world data set and use data visualization tools to communicate the insights gained from the analysis.

The data set for this project is the "Wholesale Data" dataset containing information about various products sold by a grocery store.
The project will involve the following tasks:

-	Exploratory data analysis and pre-processing: We will import and clean the data sets, analyze and visualize the relationships between the different variables, handle missing values and outliers, and perform feature engineering as needed.
-	Unsupervised learning: We will use the Wholesale Data dataset to perform k-means clustering, hierarchical clustering, and principal component analysis (PCA) to identify patterns and group similar data points together. We will determine the optimal number of clusters and communicate the insights gained through data visualization.

The ultimate goal of the project is to gain insights from the data sets and communicate these insights to stakeholders using appropriate visualizations and metrics to make informed decisions based on the business questions asked."

## Process
### 1. Exploratory data analysis and preprocessing
<p> The dataset was provided as a pdf file as part of a project to identify the clusters in a wholesale data containing columns such as:

1. Frozen
2. Fresh
3. Region 
4. Channel 
5. etc

I saved the wholesale data csv file as a variable called data <p>


<p>While exploring the data there seems like an abnormal distribution of the data, for the 
Fresh, Milk, Grocery, Frozen, Detergents_Paper & Delicassen columns because the 75% is much different from the max number for all the columns. It indicates that there is likely a large left skew.<p>

### 1b. Visualization

![Correlation Heatmap With Outliers](/images/CorrelationHeatmapWithOutliers3.png "Correlation Heatmap With Outliers")

<p>The correlation between Detergents paper and grocery is the highest at 0.92 followed by a high correlation between the grocery and Milk of 0.73, 0.66 between Milk and Detergent paper, 0.64 between detergent paper and Channel, 0.61 between grocery and Channel
</p>

</br>

![Column Pairplot with Outliers](/images/PairplotWithOutliers.png "Column Pairplot with Outliers")

<p>The distribution of the data shows an extreme right skew of the data for all the data in the columns selected. There also seems to be a lot of outliers which can explain the disparity between the mean and the max amount of these columns.
</p>

</br>

<p>Dealing with the outliers<p>

```
# Plotting a boxplot for all columns except the Region & Channel columns

grocery_columns=['Fresh', 'Milk', 'Grocery','Frozen', 'Detergents_Paper', 'Delicassen'] # subset of columns to plot

def plot_boxplot(df, columns=grocery_columns):
    for i in columns:
        sns.boxplot(
            y=data[i],
            x=data['Region'],
            hue=data['Channel'])
        plt.savefig(f'{i}BoxplotWithOutliers')
        plt.show()
```
```
# Function to replace outliers with the average of non_outlier amount
def removing_outliers(data):
    for column_name, column_data in data.items():
        if pd.api.types.is_numeric_dtype(column_data):
            Q1 = column_data.quantile(0.25)
            Q3 = column_data.quantile(0.75)

            # Calculate the IQR
            IQR = Q3 - Q1

            # Defining outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify Non-Outliers
            non_outliers = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]

            # Calculate the average of Non-Outliers Values
            average_non_outliers = int(non_outliers.mean())

            # Replace outliers with the avg of non-outliers
            data.loc[(column_data < lower_bound) | (column_data > upper_bound), column_name] = average_non_outliers

    return data
```

<p>In the boxplots showing the columns prior to the outliers being filtered out and replaced with the average amount of that column. I noticed the following:
<p>

- Channel 2 has more outliers than Channel 1
- Region 3 has more more outliers than the other Regions

<p>Channel 2 had exteme outliers in almost all the columns according the boxplot visualization of each column. After removing the outliers from each column, there seems to be more outliers in Channel 1, this might be because on aerage of the total column, the outliers on channel 1 might not be considered an outlier if they were on channel 2. But since we removed the outliers per column and not by column grouped by Channel. outliers are still showing on Channel 1 but most outliers on Channel 2 have been filtered out. 
<p>

</br>

![Column Pairplot without Outliers](/images/PairplotWithoutOutliers.png "Column Pairplot without Outliers")

<p>Unlike the Pairplot with outliers, it is harder to see the positive correlation for the scatterplots. There is still a right skew for the data, but the distribution seems better than before now that the outliers have been removed
<p>

</br>

![Correlation Heatmap Without Outliers](/images/CorrelationHeatmapWithoutOutliers.png "Correlation Heatmap With Outliers")

<p>The Correlation between the columns have reduced now that the outliers have been dropped. The correlation between Detergents paper and grocery is still the highest at 0.74, followed by Detergents paper and Channel of 0.70, 0.64 between Milk and Grocery, 0.64 between Grocery and Channel, 0.59 between milk and channel
<p>


### 2. KMeans Clustering
![KMeans number of clusters](/images/KMeansElbowNumberOfClusters.png "KMeans number of clusters")

<p> From the elbow plot, the optimal clusters is around 6 
<p>

```
Code to create plot to determine the optimal clusters using KMeans

SSE=[] #creating empty list to store the inertia score
for cluster in range(1,15):
    kmeans = KMeans(n_clusters=cluster, init='k-means++', n_init='auto')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

#converting the result into a dataframe to plot it
kmeans_df = pd.DataFrame({'Cluster':range(1,15), 'SSE': SSE})
plt.figure(figsize=(10,6))
plt.plot(kmeans_df['Cluster'], kmeans_df['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('KMeansElbowNumberOfClusters.png')
plt.show()
```

### 3. Hierarchical Clustering 

![Dendogram Plot](/images/DendogramPlot.png "Dendogram Plot")
```
# plotting a dendogram to determine the optimal number of clusters for a hierarchial cluster
plt.figure(figsize=(10,6))
dend=sch.dendrogram(sch.linkage(data_scaled, method='ward'))
plt.title('Dendogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean distances')
plt.savefig('DendogramPlot.png')
plt.show()
```
<p> Dendogram plot shows that the optimal clusters using hiereachial clustering is 2
<p>

```
# Fit and predict cluster using Agglomerative clustering and the optimal cluster gotten from the dendogram plot
cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)
```

```
#Creating cluster plot using the fitted agglomerative data and Detergents_Paper and Grocery
plt.figure(figsize=(10, 6))  
plt.scatter(data_scaled['Detergents_Paper'], data_scaled['Grocery'], c=cluster.labels_)
plt.savefig('HierarchialClusteringWithDetergents_Paper&Grocery.png')
plt.show() 
```
![Hierarchial Clustering](/images/HierarchialClusteringWithDetergents_Paper&Grocery.png "Hierarchial Clustering")


### 5. Findings

1. Prior to cleaning of the data, I noticed that the data for all columns except the Region and Channel colun were extremely left skewed. After replacing outliers with the mean of the non-outliers data there is still a left skew but it's way less than before.
2. I noticed there were more outliers in Region 3 in comparison to other regions. More information might be needed to conclude on why this might be happening.
3. While using a PCA of 2, 55% of the variance can be explained. I also noticed that Detergents_Paper, Grocery, Channel & main contributors to the PC1, which compound combinations of features best describe customers.  
4. The number of clusters for each analysis includes:
    -  PCA : It seems like there were 3 clusters, the third cluster seems like it was more of noise
    -  KMeans Clustering: The optimal number of clusters using inertia was around 6
    -  Hierachial Clustering: The number of clustering was identified at 2.
    -  Given the various results, I will like to conclude that there are 2 or 3 mainly identified clusters.