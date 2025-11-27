import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
   
    
    def __init__(self, config):
        self.config = config
        self.img_size = config.get('img_size', 32)
        
    def load_data(self, csv_path):
     
        print(f"正在加载数据: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"加载完成：{len(df)} 个样本，{df.shape[1] - 1} 个特征")
        
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        
        return features, labels, df
    
    def optimized_feature_grouping(self, features, labels, n_clusters=3):
      
        print("开始特征分组...")
        
       
        mi_scores = mutual_info_classif(features, labels)
        
        
        assert features.shape[1] >= 75, f"原始特征数不足75个，当前为{features.shape[1]}"
        sorted_indices = np.argsort(mi_scores)[::-1]  # 降序排列
        selected_indices = sorted_indices[:75]  # 取前75个
        selected_features = features[:, selected_indices]
        
       
        n_features = 75
        mi_matrix = np.zeros((n_features, n_features))
        
       
        def calc_mi(i, j):
            return mutual_info_regression(selected_features[:, [i]], selected_features[:, j])[0]
        
        for i in range(n_features):
            results = Parallel(n_jobs=-1)(
                delayed(calc_mi)(i, j)
                for j in range(i + 1, n_features)
            )
            for idx, j in enumerate(range(i + 1, n_features)):
                mi_matrix[i, j] = results[idx]
                mi_matrix[j, i] = results[idx]
        
        
        linkage_matrix = linkage(mi_matrix, 'average')
        
      
        if self.config.get('save_dendrogram', True):
            plt.figure(figsize=(15, 8))
            dendrogram(linkage_matrix)
            plt.title("Feature Hierarchical Clustering Dendrogram")
            plt.xlabel("Feature Index")
            plt.ylabel("Distance")
            plt.savefig('feature_clustering_dendrogram.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("特征聚类树已保存为 feature_clustering_dendrogram.png")
        
     
        clusters = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
        
       
        cluster_indices = [[] for _ in range(n_clusters)]
        for idx, cluster_id in enumerate(clusters):
            cluster_indices[cluster_id - 1].append(idx)
        
       
        all_features = []
        for group in cluster_indices:
            all_features.extend([(selected_indices[i], mi_scores[selected_indices[i]]) for i in group])
        all_features.sort(key=lambda x: x[1], reverse=True)
        balanced_groups = [
            [f[0] for f in all_features[i * 25: (i + 1) * 25]]
            for i in range(3)
        ]
        
       
        assert sum(len(g) for g in balanced_groups) == 75, 
        
        print("特征分组完成")
        return balanced_groups, selected_indices
    
    def group_features(self, features, labels):
      
        groups, selected_indices = self.optimized_feature_grouping(features, labels)
        assert len(groups) == 3, 
        assert all(len(g) == 25 for g in groups), 
        
        return (
            features[:, groups[0]],  
            features[:, groups[1]],  
            features[:, groups[2]],  
            selected_indices
        )
    
    def normalize_channel(self, channel):
     
        scaler = MinMaxScaler(feature_range=(0, 255))
        return scaler.fit_transform(channel).astype(np.uint8)
    
    def create_rgb_images(self, r_norm, g_norm, b_norm):
       
        print("正在生成RGB图像...")
        images = []
        img_size = self.img_size
        
        for i in range(len(r_norm)):
            
            red = r_norm[i].reshape(5, 5)
            green = g_norm[i].reshape(5, 5)
            blue = b_norm[i].reshape(5, 5)
            
            red_resized = cv2.resize(red, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            green_resized = cv2.resize(green, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            blue_resized = cv2.resize(blue, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            
          
            rgb = np.stack([red_resized, green_resized, blue_resized], axis=-1)
            images.append(rgb)
            
        return np.array(images)
    
    def save_images(self, images, labels, output_dir):
       
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        created_dirs = set()
        
        for i, (img, label) in enumerate(zip(images, labels)):
            label_dir = os.path.join(output_dir, str(label))
            if label_dir not in created_dirs:
                os.makedirs(label_dir, exist_ok=True)
                created_dirs.add(label_dir)
            
           
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(label_dir, f"sample_{i}.png"), img_bgr)
        
        print(f"图像已保存至：{output_dir}")
        return len(images)
    
    def generate_rgb_images(self, csv_path, output_dir):
        
        print("开始RGB图像生成流程...")
        
       
        features, labels, df = self.load_data(csv_path)
        
        r, g, b, selected_indices = self.group_features(features, labels)
        
       
        r_norm = self.normalize_channel(r)
        g_norm = self.normalize_channel(g) 
        b_norm = self.normalize_channel(b)
        
      
        rgb_images = self.create_rgb_images(r_norm, g_norm, b_norm)
        
        
        num_images = self.save_images(rgb_images, labels, output_dir)
        
        print(f"RGB图像生成完成！共生成 {num_images} 张图像")
        return rgb_images.shape, selected_indices