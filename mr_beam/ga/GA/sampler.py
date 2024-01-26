import numpy as np

class Sampler:
    def __init__(self, points, threshold, mode='pareto'):
        self.mode = mode
        self.seen = set()
        self.neighbors = []
        self.adj = []
        self.threshold = threshold
        
        self.points = points.copy()
        self.n = len(points[0])
        
        #Rescale
        for i in range(len(self.points)):
            self.points[i] /= np.max(self.points[i])
                 
        return

    def find_points(self, fits):
        self.points = np.zeros((fits.shape[2], fits.shape[1]))
        if self.mode == 'pareto':
            for i in range(self.points.shape[0]-1):
                self.points[i] = fits[0, :, i]-fits[0, :, -1]
            self.points[-1] = fits[0,:,-1]
        else:
            for i in range(self.points.shape[0]):
                self.points[i] = fits[0, :, i]
        
        #Rescale
        for i in range(len(self.points)):
            self.points[i] /= np.max(self.points[i])
        return self.points
    
    def find_neighbors(self):
        self.adj = []
        possible_adj = np.arange(self.n) 
        for j in range(self.n):
            decision_vector = np.zeros(self.n, dtype=bool)
            diff = np.linalg.norm(self.points.transpose()-self.points[:,j], axis=1)
            self.adj.append(possible_adj[np.where(diff<self.threshold, True, decision_vector)])                      
        return self.adj
   
    def _dfs(self, i, j):
        if i in self.seen:
            return
        self.seen.add(i)
        self.neighbors[j].append(i)
        for nb in self.adj[i]:
            self._dfs(nb, j)
   
    def find_clusters(self):
      self.neighbors = []  
      self.seen = set()
      ans = 0
      for i in range(self.n):
         if i not in self.seen:
            self.neighbors.append([]) 
            ans += 1
            self._dfs(i, ans-1)
      return ans, self.neighbors
  
    def find_ideal(self):
        ideal = np.min(self.points, axis=1)
        diff = np.linalg.norm(self.points.transpose()-ideal, axis=1)
        argmin = np.argmin(diff)
        return argmin, self.points[:,argmin]
    
    def find_accumulation(self, threshold):
        self.threshold = threshold
        adj = self.find_neighbors()
        number_of_neighbors = np.zeros(self.n)
        for i in range(self.n):
            number_of_neighbors[i] = len(adj[i])
        argmin = np.argmax(number_of_neighbors)
        return argmin, len(adj[argmin])
    
    def find_accumulation_per_cluster(self, threshold):
        self.threshold = threshold
        adj = self.find_neighbors()
        ans, neighbors = self.find_clusters()
        argmins = []
        lengths = []
        for j in range(ans):
            number_of_neighbors = np.zeros(len(neighbors[j]))
            for i in range(len(neighbors[j])):
                number_of_neighbors[i] = len(adj[neighbors[j][i]])
            argmins.append(neighbors[j][np.argmax(number_of_neighbors)])
            lengths.append(np.max(number_of_neighbors))
        return argmins, lengths
            
    
        
        
        
        
        
        
        
        
        
        
        