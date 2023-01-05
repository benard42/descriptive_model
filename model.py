import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]

x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]

dataa = {'x': [1, 2, 3, 4, 5], 'y': [2, 3, 4, 5, 6]}

dataaa = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]


# Plot the histogram
plt.hist(data)
plt.show()
plt.title('histogram')

# Plot the box plot
plt.boxplot(data)
plt.show()
plt.title('box plot')

# Plot the scatter plot
plt.scatter(x, y)
plt.show()
plt.title('scatter plot')

# Calculate the correlation
correlation = pd.DataFrame(dataa).corr()
print(correlation)

# Perform clustering
kmeans = KMeans(n_clusters=2).fit(dataaa)

# Print the clusters
print(kmeans.labels_)

