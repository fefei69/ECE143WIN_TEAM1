import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, normalize

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from utils import plot_correlations, convert_dataset, discrete_analysis, importance_analysis
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



if __name__ == '__main__':
    data = pd.read_csv('Student_Insomnia.csv')
    # remove the timestamp column
    data = data.drop('Timestamp',axis=1)

    discrete_feat = ['Year_Study', 'Gender', 'Sleep_Difficulty', 
                     'Hours_of_Sleep', 'Night_Wake', 'Concentration_Difficulty', 
                     'Fatigue', 'Missed_Classes', 'Assignment_Impact', 'Device_Use', 
                     'Caffeine_Consumption', 'Exercise_Frequency', 'Stress_Levels', 'Academic_Performance']
    
    # (1) Correlation Matrix
    data_cleaned = convert_dataset(data)
    plot_correlations(data_cleaned)

    # (2) discrete analysis
    discrete_analysis(discrete_feat, data_cleaned, features='Hours_of_Sleep')

    # (3) importance of features
    importance_analysis(data_cleaned, features='Academic_Performance')