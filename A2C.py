import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.linalg as la


class Environment:

  def __init__(self, start, goal, size):  

    self.start = start
    self.goal = goal
    self.size = size


    self.env = np.zeros(size)
    self.reward_matrix = np.full((4,size[0],size[1]),-1)
    self.reward_matrix[:, goal[0], goal[1]] = 0
    # self.reward_matrix[:, start[0], start[1]] = 0
    

    # self.blocks = [(1,0)  , (1,1), (1,2), (1,3), (3,1), (3,2), (3,3), (3,4)]
    self.blocks = [(4,4)]
    # self.createBlocks()

    self.dim = size[0]
    self.adjacency_matrix = np.zeros((self.dim ** 2, self.dim ** 2))
    self.degree_matrix = np.zeros((self.dim ** 2, self.dim ** 2))

    self.laplacian = 0
    self.eigvecs = []

    self.build_laplacian()

    self.qweights = np.zeros((1, 4 * self.dim * self.dim))
    self.qvalues = lambda s,a: (self.qweights.dot(self.get_features(s,a).T))[0][0]


  def find_neighbours(self, s):

    n = self.dim
    val = []

    if s[1] - 1 >= 0:
      s1 = (s[0], s[1] - 1)
      val.append(s1[0] * n + s1[1])

    if s[1] + 1 <= 4:
      s1 = (s[0], s[1] + 1)
      val.append(s1[0] * n + s1[1])
    
    if s[0] - 1 >= 0:
      s1 = (s[0] - 1, s[1])
      val.append(s1[0] * n + s1[1])

    if s[0] + 1 <= 4:
      s1 = (s[0] + 1, s[1])
      val.append(s1[0] * n + s1[1])
    
    return val


  def build_laplacian(self):

    for i in range(self.dim):
      for j in range(self.dim):

        s_no = i * self.dim + j
        s = (i,j)

        for k in self.find_neighbours(s):
          self.adjacency_matrix[s_no, k] = 1



    degree_arr = sum(self.adjacency_matrix)

    for i in range(self.dim ** 2):
      self.degree_matrix[i,i] = degree_arr[i]

    self.laplacian = self.degree_matrix - self.adjacency_matrix
    eigvals, self.eigvecs = la.eig(self.laplacian)


  def get_features(self, s, a):

    s_no = s[0] * self.dim + s[1]
    feature = self.eigvecs[s_no]
    arr = np.zeros((4 * self.dim * self.dim))

    k = self.dim * self.dim

    if a == 0:
      arr[:k] = feature

    if a == 1:
      arr[k:2*k] = feature

    if a == 2:
      arr[2*k:3*k] = feature

    if a == 3:
      arr[3*k: 4*k] = feature

    return np.array([arr])


  def createBlocks(self):

    for s in self.blocks:

      self.reward_matrix[:,s[0],s[1]] = -100


  def nextState(self, s,a):

    if a == 0:

      if s[1] == 0:
        return (s[0],s[1])

      return (s[0], s[1] - 1)

    elif a == 1:

      if s[1] == self.size[1] - 1:
        return (s[0], s[1])

      return (s[0], s[1] + 1)

    elif a == 2:

      if s[0] == 0:
        return (s[0], s[1])

      return (s[0] - 1, s[1])

    elif a == 3:

      if s[0] == self.size[0] - 1:
        return (s[0], s[1])

      return (s[0] + 1, s[1])

  
  def reward(self, s,a):
    return self.reward_matrix[a,s[0],s[1]]


class Agent:

  def __init__(self, size):

    self.size = size
    self.actions = ["left", "right", "up", "down"]

    self.state_policy = np.full((4, size[0], size[1]), 0.25)

class ActorCritic:

  def __init__(self, df, env, agent):

    self.df = df
    
    self.grid = env
    self.agent = agent

    self.step_size_qvalue = 0.1
    self.step_size_policy = 0.01

    self.errors = {(i,j,k): list() for i in range(self.grid.size[0]) for j in range(self.grid.size[1]) for k in range(len(self.agent.actions))}

    self.parameter = np.random.uniform(0,1,(1,4 * self.grid.dim * self.grid.dim))
    

    self.regret = [[-8., -7., -6., -5., -4.],
       [-7., -6., -5., -4., -3.],
       [-6., -5., -4., -3., -2.],
       [-5., -4., -3., -2., -1.],
       [-4., -3., -2., -1.,  0.]]
    # self.regret = [[-14., -13., -12., -11., -10.,  -9.,  -8.,  -7.],
    #    [-13., -12., -11., -10.,  -9.,  -8.,  -7.,  -6.],
    #    [-12., -11., -10.,  -9.,  -8.,  -7.,  -6.,  -5.],
    #    [-11., -10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.],
    #    [-10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.],
    #    [ -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.],
    #    [ -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.],
    #    [ -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,   0.]]
    
    self.regretError = []
    self.regfin = []

  def main(self, num_of_ep, qc, gcnphi, alph):



    for iterations in range(num_of_ep):

      tot_rew = 0
      s = (np.random.randint(0, self.grid.size[0]), np.random.randint(0, self.grid.size[1]))
      start_state = s
      a = np.random.choice([0,1,2,3], p = self.policy_model(s))
      
      while True:

        if s == self.grid.goal:
          break

        s1 = self.grid.nextState(s,a)
        r = self.grid.reward(s,a)
        tot_rew += r
        a1 = np.random.choice([0,1,2,3], p = self.policy_model(s1))

        self.parameter += self.step_size_policy * qc[a, s[0], s[1]] * self.score_fn(s,a)
        # self.parameter += self.step_size_policy * self.grid.qvalues(s,a) * self.score_fn(s,a)

        td_error1 = r + self.df * qc[a1, s1[0], s1[1]] - qc[a, s[0], s[1]]
	td_error2 = r + self.df * qc[a1, s1[0], s1[1]] - qc[a, s[0], s[1]] + self.df * gcnphi[s1] - gcnphi[s]
        # td_error = r + self.df * self.grid.qvalues(s1,a1) - self.grid.qvalues(s,a)
	td_comb = alph * td_error1 + (1 - alph) * td_error2

        self.grid.qweights += self.step_size_qvalue * td_comb * self.grid.get_features(s,a)


        self.errors[s[0], s[1], a].append(td_error)

        s = s1
        a = a1

      print("episode no ", iterations)
      self.value_fn()

      self.regretError.append(self.regret[start_state[0]][start_state[1]] - tot_rew)
      self.regfin.append(sum(self.regretError))


  def main_ac(self, num_of_ep):



    for iterations in range(num_of_ep):
      
      tot_rew = 0
      s = (np.random.randint(0, self.grid.size[0]), np.random.randint(0, self.grid.size[1]))
      start_state = s
      a = np.random.choice([0,1,2,3], p = self.policy_model(s))
      
      while True:

        if s == self.grid.goal:
          break

        s1 = self.grid.nextState(s,a)
        r = self.grid.reward(s,a)
        tot_rew += r
        a1 = np.random.choice([0,1,2,3], p = self.policy_model(s1))

        # self.parameter += self.step_size_policy * qc[a, s[0], s[1]] * self.score_fn(s,a)
        self.parameter += self.step_size_policy * self.grid.qvalues(s,a) * self.score_fn(s,a)

        # td_error = r + self.df * qc[a1, s1[0], s1[1]] - qc[a, s[0], s[1]]
        td_error = r + self.df * self.grid.qvalues(s1,a1) - self.grid.qvalues(s,a)

        self.grid.qweights += self.step_size_qvalue * td_error * self.grid.get_features(s,a)


        self.errors[s[0], s[1], a].append(td_error)

        s = s1
        a = a1

      print("episode no ", iterations)
      self.value_fn()

      self.regretError.append(self.regret[start_state[0]][start_state[1]] - tot_rew)
      self.regfin.append(sum(self.regretError))

  def policy_model(self, s):
        
      vals = []
      
      for i in range(len(self.agent.actions)):
          
        vals.append(self.grid.get_features(s,i).dot(self.parameter.T)[0][0])
          
  
      vals -= max(vals)
      
      n = np.exp(vals)
      
      vals = n/sum(n)
     
      return vals
            
        
  def score_fn(self, state, action):
          
      avg = np.zeros((1,4 * self.grid.dim * self.grid.dim))    
      
      probs = self.policy_model(state)
      
      for i in range(len(probs)):
              
              avg += self.grid.get_features(state,i) * probs[i]
      
      return self.grid.get_features(state,action) - avg
       
  def plot_error(self):

    all_state_errors = list(self.errors.values())

    for error_perstate in all_state_errors:
      plt.plot(error_perstate)

  
  def find_qvalues(self, s):

    vals = []

    for a in range(len(self.agent.actions)):
      vals.append(self.grid.qvalues(s,a))
    
    return vals


  def value_fn(self):

    world = np.zeros((self.grid.size))

    for i in range(self.grid.size[0]):

      for j in range(self.grid.size[1]):
    
        s = (i,j)

        world[s] = max(self.find_qvalues(s))

    print(world)


  def printPolicy(self):

    world = np.zeros((self.grid.size))

    for i in range(self.grid.size[0]):

      for j in range(self.grid.size[1]):
    
        s = (i,j)

        world[s] = np.argmax(self.find_qvalues(s))

    print(world)

